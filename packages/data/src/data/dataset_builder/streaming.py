"""Re-iterable, memory-bounded SID-Set stream with data-package augmentation.

`iter_sid_encoded()` streams raw SID-Set image bytes; this module wraps it so
every image is decoded, resized and (optionally) run through
`data.augmentation.ImageAugmenter` -- the six realistic corruptions from the
problem statement (JPEG compression, Gaussian blur, resize, Gaussian noise,
colour jitter, centre crop) -- before it is yielded.

Drop-in for `to_labeled_samples(load_sid_subset(...))` wherever a
`shared_types.TrainableModel.train()` / `.evaluate()` consumes an
`Iterable[LabeledImageSample]` (e.g. `NormalClassifierTrainer`), except that
the whole subset is never held in memory.

Three optimisations over a naive `iter_sid_subset()` loop:

* **Off-thread image decode** -- the source is `iter_sid_encoded()`, so the
  streaming thread never pays the PNG/JPEG decode; decoding, resizing and the
  transform chain all happen on a worker pool.
* **Parallel pipeline** -- `backend="thread"` (the default) fans decode +
  resize + transform out across worker threads, yielding results in input
  order with a bounded in-flight window. Threads (not processes) are the
  default because Pillow/NumPy release the GIL, so there is real speed-up
  without pickling every image; ``backend="process"`` is available for
  many-core boxes.
* **Local disk cache** -- pass ``cache_dir`` and the first pass writes each
  image's raw bytes to disk as it streams; every later pass (``evaluate()``
  after ``train()``, a re-run, a second notebook) reads bytes from disk and
  never touches the network. Decode + augmentation still run fresh each pass.
"""

from __future__ import annotations

import json
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterator

from PIL import Image
from shared_types import LabeledImageSample, SourceMetadata

from data.augmentation import ImageAugmenter

from .sid import LABEL_NAMES, iter_sid_encoded

_METADATA_KEYS = ("img_id", "sid_label", "label_name", "binary_aigc_label")


def _meta_from_entry(entry: dict) -> SourceMetadata:
    return {key: entry[key] for key in _METADATA_KEYS if key in entry}  # type: ignore[return-value]


def _decode(data: bytes, output_size: tuple[int, int] | None) -> Image.Image:
    with Image.open(BytesIO(data)) as opened:
        image = opened.convert("RGB")
    if output_size is not None:
        image = image.resize(output_size, Image.Resampling.LANCZOS)
    return image


def _threaded_pairs(
    pairs: Iterator[tuple[bytes, SourceMetadata]],
    fn: Callable[[bytes], Image.Image],
    workers: int | None,
    window: int,
) -> Iterator[tuple[Image.Image, SourceMetadata]]:
    """Apply ``fn`` to the bytes of each ``(bytes, metadata)`` pair on a thread
    pool, yielding ``(result, metadata)`` in input order with at most ``window``
    tasks in flight."""
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending: deque[tuple] = deque()
        for data, meta in pairs:
            pending.append((pool.submit(fn, data), meta))
            if len(pending) >= window:
                future, done_meta = pending.popleft()
                yield future.result(), done_meta
        while pending:
            future, done_meta = pending.popleft()
            yield future.result(), done_meta


class AugmentedSIDDataset:
    """A re-iterable stream of (optionally augmented) SID-Set samples.

    Takes the same positional arguments as `load_sid_subset()`, plus
    augmentation, parallelism and caching controls. Every `iter(dataset)`
    produces the subset again -- from the local cache if one is warm,
    otherwise by re-streaming Hugging Face. The pull is seed-deterministic,
    so every pass sees the same source images in the same order (and the
    same corruptions, since the augmenter is re-seeded from `seed`).

    Peak memory is the in-flight pipeline window plus PIL buffers -- the
    subset itself is never materialised as a list.

    Args:
        images_per_label: balanced count per SID label (real / synthetic /
            tampered), so the stream yields ``3 * images_per_label`` samples.
        seed, buffer_size, split, hf_token: forwarded to `iter_sid_encoded()`.
        augment: when False, images are only decoded + resized -- use this for
            a validation or test stream you want left clean.
        num_augmentations: how many of the six transforms to chain onto each
            image. A fixed int (1-6) applies that many to every image; an
            inclusive ``(min, max)`` pair draws a random count per image.
        output_size: (width, height) to resize every image to; None keeps
            native resolution.
        backend: ``"thread"`` (default) or ``"process"`` runs decode + resize
            + transform on worker pools; ``"sequential"`` does it inline.
        num_workers: pool size for the parallel backends (default: CPU count).
        prefetch: max items in flight for the parallel backends
            (default: ``num_workers``, min 4). Bounds memory.
        progress: show a `tqdm` bar over the iteration (default True),
            reporting throughput (img/s) and the source (HF stream / disk
            cache). A cold HF stream sits at 0 while the first ``buffer_size``
            images fill the shuffle buffer.
        cache_dir: when set, the first pass writes each image's raw bytes
            under ``cache_dir/<key>/`` and later passes read from there instead
            of Hugging Face. The key encodes split, counts, seed and shuffle
            buffer (not output_size -- the cache holds original bytes). Not
            safe for concurrent writers.
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
        self.progress = progress
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

    # --- cache ----------------------------------------------------------

    @property
    def cache_path(self) -> Path | None:
        """Directory this dataset's byte cache lives in, or None if uncached."""
        if self.cache_dir is None:
            return None
        key = f"sid-{self.split}-{self.images_per_label}pl-seed{self.seed}-buf{self.buffer_size}-raw"
        return self.cache_dir / key

    def cache_is_warm(self) -> bool:
        path = self.cache_path
        return path is not None and (path / "_complete.json").is_file()

    def warm_cache(self) -> Path:
        """Populate the disk cache: one streaming pass that just downloads and
        writes raw bytes (no decode). Returns the cache directory; a no-op if
        the cache is already warm."""
        if self.cache_dir is None:
            raise ValueError("warm_cache() needs cache_dir to be set")
        if not self.cache_is_warm():
            pairs: Iterator = self._encoded_pairs()
            if self.progress:
                pairs = self._with_bar(pairs, "caching")
            for _ in pairs:
                pass
        return self.cache_path  # type: ignore[return-value]

    def _encoded_pairs(self) -> Iterator[tuple[bytes, SourceMetadata]]:
        """Yield ``(raw encoded image bytes, metadata)`` -- from the warm disk
        cache if there is one, else from Hugging Face, writing the bytes to the
        cache as they stream when ``cache_dir`` is set."""
        cache = self.cache_path

        if self.cache_is_warm():
            assert cache is not None
            with (cache / "metadata.jsonl").open(encoding="utf-8") as manifest:
                for line in manifest:
                    entry = json.loads(line)
                    yield (cache / entry["file"]).read_bytes(), _meta_from_entry(entry)
            return

        writing = cache is not None
        manifest_file = None
        count, completed = 0, False
        if writing:
            assert cache is not None
            cache.mkdir(parents=True, exist_ok=True)
            manifest_file = (cache / "metadata.jsonl").open("w", encoding="utf-8")
        try:
            for data, metadata in iter_sid_encoded(
                self.images_per_label,
                seed=self.seed,
                buffer_size=self.buffer_size,
                split=self.split,
                hf_token=self.hf_token,
            ):
                if writing:
                    filename = f"{count:06d}.bin"
                    (cache / filename).write_bytes(data)
                    manifest_file.write(json.dumps({"file": filename, **metadata}) + "\n")
                count += 1
                yield data, metadata
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
                        }),
                        encoding="utf-8",
                    )

    # --- iteration ----------------------------------------------------

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

    def _with_bar(self, it: Iterator, verb: str) -> Iterator:
        """Wrap an iterator in a tqdm bar; a no-op if tqdm is missing."""
        try:
            from tqdm.auto import tqdm
        except ImportError:
            return it
        warm = self.cache_is_warm()
        source = "disk cache" if warm else "HF stream"
        bar = tqdm(it, total=len(self), unit="img", desc=f"SID {self.split} ({verb}, {source})")
        if not warm:
            fill = min(self.buffer_size, len(self))
            hint = " Pass a smaller buffer_size= to shorten this." if fill > 16 else ""
            bar.write(
                f"  {self.split}: streaming from HuggingFace -- the first ~{fill} images fill the "
                f"shuffle buffer before the bar moves.{hint}"
            )
        return bar

    def __iter__(self) -> Iterator[LabeledImageSample]:
        samples = self._iter_samples()
        if self.progress:
            samples = self._with_bar(samples, "augment" if self.augment else "load")
        yield from samples

    def _iter_samples(self) -> Iterator[LabeledImageSample]:
        pairs = self._encoded_pairs()
        window = self.prefetch or self.num_workers or 4

        # augment=False: decode (+ resize) only.
        if not self.augment:
            if self.backend == "sequential" or self.num_workers == 1:
                for data, metadata in pairs:
                    yield LabeledImageSample(image=_decode(data, self.output_size), metadata=metadata)
            else:
                decode = lambda data: _decode(data, self.output_size)  # noqa: E731
                for image, metadata in _threaded_pairs(pairs, decode, self.num_workers, window):
                    yield LabeledImageSample(image=image, metadata=metadata)
            return

        # augment=True: decode + resize + transform chain. The augmenter takes
        # bytes directly (via image_io.load_rgb), so the whole chain -- decode
        # included -- runs on the worker.
        augmenter = ImageAugmenter(output_size=self.output_size, seed=self.seed)

        if self.backend == "sequential" or self.num_workers == 1:
            for data, metadata in pairs:
                array, _record = augmenter.transform_one(data, self.num_augmentations)
                yield LabeledImageSample(image=Image.fromarray(array), metadata=metadata)
            return

        # iter_transform_images() pulls bytes (order-preserving, bounded
        # in-flight); shadow it with a metadata queue popped in result order.
        pending_meta: deque[SourceMetadata] = deque()

        def byte_stream() -> Iterator[bytes]:
            for data, metadata in pairs:
                pending_meta.append(metadata)
                yield data

        for array, _record in augmenter.iter_transform_images(
            byte_stream(),
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
    `to_labeled_samples(load_sid_subset(...))` that decodes off the streaming
    thread, applies the data package's `ImageAugmenter` transforms in
    parallel, and can cache the raw source bytes to local disk::

        train = augmented_sid_dataset(1000, split="train", output_size=(224, 224),
                                      cache_dir="sid_cache")
        val = augmented_sid_dataset(200, split="validation", augment=False,
                                    output_size=(224, 224), cache_dir="sid_cache")
        train.warm_cache(); val.warm_cache()   # one download pass, with a bar
        trainer.train(train, val_samples=val)  # reads bytes from the cache
        trainer.evaluate(val)
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
