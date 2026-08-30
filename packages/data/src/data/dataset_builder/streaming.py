"""Re-iterable, memory-bounded SID-Set stream with data-package augmentation.

`iter_sid_subset()` streams raw SID-Set samples; this module wraps it so every
image is run through `data.augmentation.ImageAugmenter` -- the six realistic
corruptions from the problem statement (JPEG compression, Gaussian blur,
resize, Gaussian noise, colour jitter, centre crop) -- before it is yielded.

Drop-in for `to_labeled_samples(load_sid_subset(...))` wherever a
`shared_types.TrainableModel.train()` / `.evaluate()` consumes an
`Iterable[LabeledImageSample]` (e.g. `NormalClassifierTrainer`), except that
the whole subset is never held in memory.

Two optimisations over a naive stream:

* **Parallel augmentation** -- `backend="thread"` (the default) fans the
  per-image transform chain out across worker threads via
  `ImageAugmenter.iter_transform_images()`, yielding results in input order
  with a bounded in-flight window. Threads (not processes) are the default
  because the transforms are Pillow/NumPy calls that release the GIL, so
  there is real speed-up without paying to pickle every image to a worker;
  ``backend="process"`` is available for CPU-bound cases on many cores.
* **Local disk cache** -- pass ``cache_dir`` and the first iteration writes
  each clean (resized) image + its label to disk as it streams from Hugging
  Face; every later iteration (``evaluate()`` after ``train()``, a re-run, a
  second notebook) reads those files on demand and never touches the
  network. Augmentation still happens fresh on each read.
"""

from __future__ import annotations

import json
import os
from collections import deque
from pathlib import Path
from typing import Iterator

from PIL import Image
from shared_types import LabeledImageSample, SourceMetadata

from data.augmentation import ImageAugmenter

from .sid import LABEL_NAMES, iter_sid_subset

_METADATA_KEYS = ("img_id", "sid_label", "label_name", "binary_aigc_label")


def _meta_from_entry(entry: dict) -> SourceMetadata:
    return {key: entry[key] for key in _METADATA_KEYS if key in entry}  # type: ignore[return-value]


class AugmentedSIDDataset:
    """A re-iterable stream of (optionally augmented) SID-Set samples.

    Takes the same positional arguments as `load_sid_subset()`, plus
    augmentation, parallelism and caching controls. Every `iter(dataset)`
    produces the subset again -- from the local cache if one is warm,
    otherwise by re-streaming Hugging Face. The pull is seed-deterministic,
    so every pass sees the same source images in the same order (and the
    same corruptions, since the augmenter is re-seeded from `seed`).

    Peak memory is the in-flight augmentation window plus PIL buffers -- the
    subset itself is never materialised as a list.

    Args:
        images_per_label: balanced count per SID label (real / synthetic /
            tampered), so the stream yields ``3 * images_per_label`` samples.
        seed, buffer_size, split, hf_token: forwarded to `iter_sid_subset()`.
        augment: when False, images are passed through untouched (only
            resized if ``output_size`` is set) -- use this for a validation
            or test stream you want left clean.
        num_augmentations: how many of the six transforms to chain onto each
            image. A fixed int (1-6) applies that many to every image; an
            inclusive ``(min, max)`` pair draws a random count per image.
        output_size: (width, height) to resize every image to before
            augmenting; None keeps native resolution.
        backend: ``"thread"`` (default) or ``"process"`` runs the transform
            chain on worker pools; ``"sequential"`` does it inline. Ignored
            when ``augment`` is False. Threads win here because the transforms
            release the GIL and processes must pickle every image.
        num_workers: pool size for the parallel backends (default: CPU count).
        prefetch: max transforms in flight for the parallel backends
            (default: ``num_workers``). Bounds memory.
        progress: show a `tqdm` bar over the iteration (default True). The bar
            reports throughput (img/s) and whether the source is the HF stream
            or the disk cache; a cold HF stream sits at 0 while the first
            ``buffer_size`` images fill the shuffle buffer.
        cache_dir: when set, the first iteration caches each clean resized
            image + label under ``cache_dir/<key>/`` and later iterations
            read from there instead of Hugging Face. The key encodes split,
            counts, seed, shuffle buffer and output size, so changing any of
            those uses a fresh cache. Cached images are re-encoded as JPEG
            q95. Not safe for concurrent writers.
    """

    def __init__(
        self,
        images_per_label: int,
        seed: int = 4,
        buffer_size: int = 100,
        split: str = "train",
        hf_token: str | None = None,
        *,
        augment: bool = True,
        num_augmentations: int | tuple[int, int] = 6,
        output_size: tuple[int, int] | None = None,
        backend: str = "thread",
        num_workers: int | None = None,
        prefetch: int | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        progress: bool = True,
    ) -> None:
        if images_per_label < 1:
            raise ValueError("images_per_label must be at least 1")
        if backend not in {"process", "thread", "sequential"}:
            raise ValueError("backend must be 'process', 'thread' or 'sequential'")
        self.progress = progress
        self.images_per_label = images_per_label
        self.seed = seed
        self.buffer_size = buffer_size
        self.split = split
        self.hf_token = hf_token
        self.augment = augment
        self.num_augmentations = num_augmentations
        self.output_size = output_size
        self.backend = backend
        self.num_workers = num_workers
        self.prefetch = prefetch
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    # --- cache -------------------------------------------------------------

    @property
    def cache_path(self) -> Path | None:
        """Directory this dataset's cache lives in, or None if uncached."""
        if self.cache_dir is None:
            return None
        size = f"{self.output_size[0]}x{self.output_size[1]}" if self.output_size else "native"
        key = f"sid-{self.split}-{self.images_per_label}pl-seed{self.seed}-buf{self.buffer_size}-{size}"
        return self.cache_dir / key

    def cache_is_warm(self) -> bool:
        path = self.cache_path
        return path is not None and (path / "_complete.json").is_file()

    def warm_cache(self) -> Path:
        """Populate the disk cache (a full streaming pass) without augmenting.
        Returns the cache directory. A no-op if the cache is already warm."""
        if self.cache_dir is None:
            raise ValueError("warm_cache() needs cache_dir to be set")
        if not self.cache_is_warm():
            clean = self._iter_clean()
            if self.progress:
                try:
                    from tqdm.auto import tqdm

                    clean = tqdm(clean, total=len(self), unit="img", desc=f"caching SID {self.split}")
                except ImportError:
                    pass
            for _ in clean:
                pass
        return self.cache_path  # type: ignore[return-value]

    def _iter_clean(self) -> Iterator[tuple[Image.Image, SourceMetadata]]:
        """Yield (clean RGB image at output_size, metadata), from the warm
        disk cache if there is one, else from Hugging Face -- populating the
        cache as it streams when cache_dir is set."""
        cache = self.cache_path

        if self.cache_is_warm():
            assert cache is not None
            with (cache / "metadata.jsonl").open(encoding="utf-8") as manifest:
                for line in manifest:
                    entry = json.loads(line)
                    with Image.open(cache / entry["file"]) as opened:
                        image = opened.convert("RGB")
                    yield image, _meta_from_entry(entry)
            return

        writing = cache is not None
        manifest_file = None
        count, completed = 0, False
        if writing:
            assert cache is not None
            cache.mkdir(parents=True, exist_ok=True)
            manifest_file = (cache / "metadata.jsonl").open("w", encoding="utf-8")
        try:
            for sample in iter_sid_subset(
                self.images_per_label,
                seed=self.seed,
                buffer_size=self.buffer_size,
                split=self.split,
                hf_token=self.hf_token,
            ):
                image = sample.image
                if self.output_size is not None:
                    image = image.resize(self.output_size, Image.Resampling.LANCZOS)
                if writing:
                    filename = f"{count:06d}.jpg"
                    image.save(cache / filename, format="JPEG", quality=95)
                    manifest_file.write(json.dumps({"file": filename, **sample.metadata}) + "\n")
                count += 1
                yield image, sample.metadata
            completed = True
        finally:
            if manifest_file is not None:
                manifest_file.close()
                if completed:
                    (cache / "_complete.json").write_text(
                        json.dumps({
                            "count": count,
                            "images_per_label": self.images_per_label,
                            "split": self.split,
                            "seed": self.seed,
                            "buffer_size": self.buffer_size,
                            "output_size": list(self.output_size) if self.output_size else None,
                        }),
                        encoding="utf-8",
                    )

    # --- iteration -------------------------------------------------------

    def __len__(self) -> int:
        """Known up front: the stream is balanced with `images_per_label`
        samples per SID label. Handy for progress bars; if the underlying
        stream can't fill the subset, iteration raises rather than returning
        fewer than this."""
        return self.images_per_label * len(LABEL_NAMES)

    def __repr__(self) -> str:
        n = self.num_augmentations
        mode = f"augment {n if isinstance(n, int) else f'{n[0]}-{n[1]}'} ({self.backend})" if self.augment else "clean"
        cache = "" if self.cache_dir is None else f", cache={'warm' if self.cache_is_warm() else 'cold'}"
        return (
            f"AugmentedSIDDataset(split={self.split!r}, per_label={self.images_per_label}, "
            f"{mode}, output_size={self.output_size}{cache})"
        )

    def __iter__(self) -> Iterator[LabeledImageSample]:
        if not self.progress:
            yield from self._iter_samples()
            return

        try:
            from tqdm.auto import tqdm
        except ImportError:
            yield from self._iter_samples()
            return

        warm = self.cache_is_warm()
        source = "disk cache" if warm else "HF stream"
        bar = tqdm(total=len(self), unit="img", desc=f"SID {self.split} ({source})")
        if not warm:
            fill = min(self.buffer_size, len(self))
            hint = " Pass a smaller buffer_size= (or set cache_dir=) to shorten this." if fill > 16 else ""
            bar.write(
                f"  {self.split}: streaming from HuggingFace -- the first ~{fill} images fill the "
                f"shuffle buffer before the bar moves.{hint}"
            )
        try:
            for sample in self._iter_samples():
                yield sample
                bar.update(1)
        finally:
            bar.close()

    def _iter_samples(self) -> Iterator[LabeledImageSample]:
        clean = self._iter_clean()

        if not self.augment:
            for image, metadata in clean:
                yield LabeledImageSample(image=image, metadata=metadata)
            return

        # Images arrive already resized, so the augmenter does resize=None.
        augmenter = ImageAugmenter(output_size=None, seed=self.seed)

        if self.backend == "sequential" or self.num_workers == 1:
            for image, metadata in clean:
                array, _record = augmenter.transform_one(image, self.num_augmentations)
                yield LabeledImageSample(image=Image.fromarray(array), metadata=metadata)
            return

        # Parallel: iter_transform_images() pulls images (order-preserving,
        # bounded in-flight); we shadow it with a metadata queue popped in the
        # same order the results come back.
        pending_meta: deque[SourceMetadata] = deque()

        def images() -> Iterator[Image.Image]:
            for image, metadata in clean:
                pending_meta.append(metadata)
                yield image

        for array, _record in augmenter.iter_transform_images(
            images(),
            self.num_augmentations,
            return_metadata=True,
            backend=self.backend,
            num_workers=self.num_workers,
            prefetch=self.prefetch,
        ):
            yield LabeledImageSample(image=Image.fromarray(array), metadata=pending_meta.popleft())


def augmented_sid_dataset(
    images_per_label: int,
    seed: int = 4,
    buffer_size: int = 100,
    split: str = "train",
    hf_token: str | None = None,
    *,
    augment: bool = True,
    num_augmentations: int | tuple[int, int] = 6,
    output_size: tuple[int, int] | None = None,
    backend: str = "thread",
    num_workers: int | None = None,
    prefetch: int | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    progress: bool = True,
) -> AugmentedSIDDataset:
    """Return a re-iterable `AugmentedSIDDataset` (see that class for details).

    Same positional signature as `load_sid_subset()`; the return value is a
    streaming, memory-bounded stand-in for
    `to_labeled_samples(load_sid_subset(...))` that also applies the data
    package's `ImageAugmenter` transforms -- in parallel by default -- and
    can cache the source pull to local disk::

        train = augmented_sid_dataset(1000, split="train", output_size=(224, 224),
                                      cache_dir="sid_cache")
        val = augmented_sid_dataset(200, split="validation", augment=False,
                                    output_size=(224, 224), cache_dir="sid_cache")
        trainer.train(train, val_samples=val)   # first pass fills the cache
        trainer.evaluate(val)                   # reads the cache, no network
    """
    return AugmentedSIDDataset(
        images_per_label,
        seed=seed,
        buffer_size=buffer_size,
        split=split,
        hf_token=hf_token,
        augment=augment,
        num_augmentations=num_augmentations,
        output_size=output_size,
        backend=backend,
        num_workers=num_workers,
        prefetch=prefetch,
        cache_dir=cache_dir,
        progress=progress,
    )
