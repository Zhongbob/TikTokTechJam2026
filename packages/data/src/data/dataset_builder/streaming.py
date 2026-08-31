"""Re-iterable, memory-bounded HF image streams with data-package augmentation.

A source yields ``(raw encoded image bytes, metadata)`` one at a time
(`sid.iter_sid_encoded`, `wildfake.iter_wildfake_encoded`, ...). This module
wraps such a source so every image is decoded, resized and (optionally) run
through `data.augmentation.ImageAugmenter` -- the six realistic corruptions from
the problem statement (JPEG compression, Gaussian blur, resize, Gaussian noise,
colour jitter, centre crop) -- before it is yielded as a
`shared_types.LabeledImageSample`.

`StreamingAugmentedDataset` is the generic engine; `AugmentedSIDDataset` /
`augmented_sid_dataset()` wire it to SID-Set. It is a drop-in for
`to_labeled_samples(load_sid_subset(...))` wherever a
`shared_types.TrainableModel.train()` / `.evaluate()` consumes an
`Iterable[LabeledImageSample]`, except that the whole subset is never held in
memory.

Three optimisations over a naive per-image loop:

* **Off-thread image decode** -- the source yields bytes, so the streaming
  thread never pays the PNG/JPEG decode; decoding, resizing and the transform
  chain all happen on a worker pool.
* **Parallel pipeline** -- ``backend="thread"`` (the default) fans decode +
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
from shared_types import AugmentationRecord, ImagePairSample, LabeledImageSample, SourceMetadata

from data.augmentation import ImageAugmenter

from .sid import LABEL_NAMES, iter_sid_encoded

EncodedSource = Callable[[], Iterator[tuple[bytes, SourceMetadata]]]


def _meta_from_entry(entry: dict) -> SourceMetadata:
    """Reconstruct a sample's metadata from a cache manifest line (everything
    except the private ``file`` pointer)."""
    return {key: value for key, value in entry.items() if key != "file"}  # type: ignore[return-value]


def _variant_meta(metadata: SourceMetadata, variant: int, total: int) -> SourceMetadata:
    """Tag a sample with its augmentation-variant index when the dataset emits
    more than one variant per source image; otherwise pass metadata through."""
    if total <= 1:
        return metadata
    return {**metadata, "variant": variant}  # type: ignore[misc]


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


class StreamingAugmentedDataset:
    """Re-iterable stream of (optionally augmented) samples from a byte source.

    ``encoded_source`` is a zero-arg callable returning a FRESH
    ``Iterator[(bytes, SourceMetadata)]`` each call (so the dataset can be
    iterated more than once). ``total`` is the number of source images it
    yields, ``cache_key`` names this stream's on-disk byte cache.

    Every ``iter(dataset)`` reproduces the stream -- from the local cache if one
    is warm, otherwise by re-running ``encoded_source``. Peak memory is the
    in-flight pipeline window plus PIL buffers; the stream is never materialised
    as a list.

    Args:
        encoded_source, total, cache_key: see above.
        desc: label for the progress bar / repr.
        seed: re-seeds the `ImageAugmenter` so corruptions are reproducible
            across passes; also used in bar messaging.
        buffer_size: source shuffle-buffer size, for bar messaging only.
        augment: when False, images are only decoded + resized -- use this for a
            validation or test stream you want left clean.
        num_augmentations: how many of the six transforms to chain onto *one*
            output image. A fixed int (0-6) applies that many to every image; an
            inclusive ``(min, max)`` pair draws a random count per image. ``0``
            (or a range starting at 0, e.g. ``(0, 6)``) leaves some images clean.
        variants_per_image: how many differently-augmented output images each
            source image produces (dataset expansion). Requires ``augment=True``;
            variant index is written to ``metadata["variant"]`` when > 1.
        output_size: (width, height) to resize every image to; None keeps native
            resolution.
        backend: ``"thread"`` (default) / ``"process"`` run decode + resize +
            transform on worker pools; ``"sequential"`` does it inline.
        num_workers: pool size for the parallel backends (default: CPU count).
        prefetch: max items in flight for the parallel backends
            (default: ``num_workers``, min 4). Bounds memory.
        cache_dir: when set, the first pass writes each image's raw bytes under
            ``cache_dir/<cache_key>/`` and later passes read from there instead
            of the network. Not safe for concurrent writers.
        progress: show a `tqdm` bar over the iteration (default True).
    """

    def __init__(
        self,
        *,
        encoded_source: EncodedSource,
        total: int,
        cache_key: str,
        desc: str = "dataset",
        seed: int = 4,
        buffer_size: int = 100,
        augment: bool = True,
        num_augmentations: int | tuple[int, int] = 6,
        variants_per_image: int = 1,
        output_size: tuple[int, int] | None = None,
        backend: str = "thread",
        num_workers: int | None = None,
        prefetch: int | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        progress: bool = True,
    ) -> None:
        if total < 1:
            raise ValueError("total must be at least 1")
        if backend not in {"process", "thread", "sequential"}:
            raise ValueError("backend must be 'process', 'thread' or 'sequential'")
        if variants_per_image < 1:
            raise ValueError("variants_per_image must be at least 1")
        if variants_per_image > 1 and not augment:
            raise ValueError("variants_per_image > 1 needs augment=True (clean variants are identical)")

        self._encoded_source = encoded_source
        self._total = total
        self._cache_key = cache_key
        self._desc = desc
        self.seed = seed
        self.buffer_size = buffer_size
        self.augment = augment
        self.num_augmentations = num_augmentations
        self.variants_per_image = variants_per_image
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
        return self.cache_dir / self._cache_key

    def cache_is_warm(self) -> bool:
        path = self.cache_path
        return path is not None and (path / "_complete.json").is_file()

    @property
    def _source_count(self) -> int:
        """Number of distinct source images (before `variants_per_image`)."""
        return self._total

    def warm_cache(self) -> Path:
        """Populate the disk cache: one streaming pass that just downloads and
        writes raw bytes (no decode). Returns the cache directory; a no-op if
        the cache is already warm."""
        if self.cache_dir is None:
            raise ValueError("warm_cache() needs cache_dir to be set")
        if not self.cache_is_warm():
            pairs: Iterator = self._encoded_pairs()
            if self.progress:
                pairs = self._with_bar(pairs, "caching", total=self._source_count)
            for _ in pairs:
                pass
        return self.cache_path  # type: ignore[return-value]

    def _encoded_pairs(self) -> Iterator[tuple[bytes, SourceMetadata]]:
        """Yield ``(raw encoded image bytes, metadata)`` -- from the warm disk
        cache if there is one, else from ``encoded_source``, writing the bytes
        to the cache as they stream when ``cache_dir`` is set."""
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
            for data, metadata in self._encoded_source():
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
                        json.dumps({"count": count, "cache_key": self._cache_key}),
                        encoding="utf-8",
                    )

    # --- iteration ----------------------------------------------------

    def __len__(self) -> int:
        """Expected sample count: `total` source images times
        `variants_per_image`."""
        return self._total * self.variants_per_image

    def __repr__(self) -> str:
        n = self.num_augmentations
        mode = (f"augment {n if isinstance(n, int) else f'{n[0]}-{n[1]}'} ({self.backend})"
                if self.augment else "clean")
        variants = f", x{self.variants_per_image} variants" if self.variants_per_image > 1 else ""
        cache = "" if self.cache_dir is None else f", cache={'warm' if self.cache_is_warm() else 'cold'}"
        return (f"{type(self).__name__}({self._desc!r}, n={self._total}, "
                f"{mode}{variants}, output_size={self.output_size}{cache})")

    def _with_bar(self, it: Iterator, verb: str, total: int | None = None) -> Iterator:
        """Wrap an iterator in a tqdm bar; a no-op if tqdm is missing."""
        try:
            from tqdm.auto import tqdm
        except ImportError:
            return it
        total = len(self) if total is None else total
        warm = self.cache_is_warm()
        source = "disk cache" if warm else "HF stream"
        bar = tqdm(it, total=total, unit="img", desc=f"{self._desc} ({verb}, {source})")
        if not warm:
            fill = min(self.buffer_size, total)
            hint = " Pass a smaller buffer_size= to shorten this." if fill > 16 else ""
            bar.write(
                f"  {self._desc}: streaming from HuggingFace -- the first ~{fill} images "
                f"fill the shuffle buffer before the bar moves.{hint}"
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
        # included -- runs on the worker. Each source image is emitted
        # `variants_per_image` times; the augmenter's RNG advances per call, so
        # the variants get different (but seed-reproducible) transform chains.
        augmenter = ImageAugmenter(output_size=self.output_size, seed=self.seed)
        variants = self.variants_per_image

        if self.backend == "sequential" or self.num_workers == 1:
            for data, metadata in pairs:
                for v in range(variants):
                    array, _record = augmenter.transform_one(data, self.num_augmentations)
                    yield LabeledImageSample(image=Image.fromarray(array), metadata=_variant_meta(metadata, v, variants))
            return

        # iter_transform_images() pulls bytes (order-preserving, bounded
        # in-flight); shadow it with a metadata queue popped in result order.
        pending_meta: deque[SourceMetadata] = deque()

        def byte_stream() -> Iterator[bytes]:
            for data, metadata in pairs:
                for v in range(variants):
                    pending_meta.append(_variant_meta(metadata, v, variants))
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


class AugmentedSIDDataset(StreamingAugmentedDataset):
    """`StreamingAugmentedDataset` wired to a balanced `saberzl/SID_Set` subset.

    Takes the same positional arguments as `load_sid_subset()`, plus
    augmentation, parallelism and caching controls. Iterating yields
    ``3 * images_per_label * variants_per_image`` `LabeledImageSample`s (real /
    synthetic / tampered), re-streaming HF (or reading the warm cache) each pass.

    Args:
        images_per_label: balanced count per SID label.
        seed, buffer_size, split, hf_token: forwarded to `iter_sid_encoded()`.
        (see `StreamingAugmentedDataset` for augment / num_augmentations /
        variants_per_image / output_size / backend / num_workers / prefetch /
        cache_dir / progress.)
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
        variants_per_image: int = 1,
        output_size: tuple[int, int] | None = None,
        backend: str = "thread",
        num_workers: int | None = None,
        prefetch: int | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        progress: bool = True,
    ) -> None:
        if images_per_label < 1:
            raise ValueError("images_per_label must be at least 1")
        self.images_per_label = images_per_label
        self.split = split
        self.hf_token = hf_token
        super().__init__(
            encoded_source=lambda: iter_sid_encoded(
                images_per_label, seed=seed, buffer_size=buffer_size, split=split, hf_token=hf_token,
            ),
            total=images_per_label * len(LABEL_NAMES),
            cache_key=f"sid-{split}-{images_per_label}pl-seed{seed}-buf{buffer_size}-raw",
            desc=f"SID {split}",
            seed=seed,
            buffer_size=buffer_size,
            augment=augment,
            num_augmentations=num_augmentations,
            variants_per_image=variants_per_image,
            output_size=output_size,
            backend=backend,
            num_workers=num_workers,
            prefetch=prefetch,
            cache_dir=cache_dir,
            progress=progress,
        )

    def __repr__(self) -> str:
        n = self.num_augmentations
        mode = (f"augment {n if isinstance(n, int) else f'{n[0]}-{n[1]}'} ({self.backend})"
                if self.augment else "clean")
        variants = f", x{self.variants_per_image} variants" if self.variants_per_image > 1 else ""
        cache = "" if self.cache_dir is None else f", cache={'warm' if self.cache_is_warm() else 'cold'}"
        return (
            f"AugmentedSIDDataset(split={self.split!r}, per_label={self.images_per_label}, "
            f"{mode}{variants}, output_size={self.output_size}{cache})"
        )


class AutoencoderDataset(AugmentedSIDDataset):
    """Stream clean + augmented image pairs for augmentation-reversal training.

    Each source SID image yields:

    * one identity pair: ``input_image == target_image == original clean image``
    * one or more augmented pairs: ``input_image = transformed image``,
      ``target_image = same clean original image``

    This keeps the paired target aligned to the original, unaugmented source,
    while still exposing the augmented inputs that the autoencoder must undo.
    """

    def __len__(self) -> int:
        return self.images_per_label * len(LABEL_NAMES) * (1 + self.variants_per_image)

    def __repr__(self) -> str:
        n = self.num_augmentations
        mode = f"augment {n if isinstance(n, int) else f'{n[0]}-{n[1]}'} ({self.backend})" if self.augment else "clean"
        variants = f", x{self.variants_per_image} variants" if self.variants_per_image > 1 else ""
        cache = "" if self.cache_dir is None else f", cache={'warm' if self.cache_is_warm() else 'cold'}"
        return (
            f"AutoencoderDataset(split={self.split!r}, per_label={self.images_per_label}, "
            f"{mode}{variants}, output_size={self.output_size}{cache})"
        )

    def __iter__(self) -> Iterator[ImagePairSample]:
        samples = self._iter_samples()
        if self.progress:
            samples = self._with_bar(samples, "autoencoder")
        yield from samples

    def _iter_samples(self) -> Iterator[ImagePairSample]:
        augmenter = ImageAugmenter(output_size=self.output_size, seed=self.seed)

        def _target_for_record(clean: Image.Image, record: AugmentationRecord | None) -> Image.Image:
            if record is None:
                return clean.copy()
            for step in record.parameters.get("steps", []):
                if step["transform"] == "center_crop":
                    crop_ratio = step["parameters"]["crop_ratio"]
                    width, height = clean.size
                    crop_width = max(1, round(width * crop_ratio))
                    crop_height = max(1, round(height * crop_ratio))
                    left = (width - crop_width) // 2
                    top = (height - crop_height) // 2
                    return clean.crop((left, top, left + crop_width, top + crop_height)).resize(
                        (width, height), Image.Resampling.LANCZOS
                    )
            return clean.copy()

        for data, metadata in self._encoded_pairs():
            clean = _decode(data, self.output_size)
            clean_copy = clean.copy()
            source = str(metadata.get("img_id") or metadata.get("label_name") or "unknown-source")
            identity = AugmentationRecord(source=source, transform="identity", parameters={})
            yield ImagePairSample(input_image=clean_copy.copy(), target_image=clean_copy.copy(), record=identity)

            if not self.augment:
                continue

            arrays, records = augmenter.transform_images(
                [data] * self.variants_per_image,
                self.num_augmentations,
                return_metadata=True,
                backend=self.backend,
                num_workers=self.num_workers,
            )
            for array, record in zip(arrays, records):
                yield ImagePairSample(
                    input_image=Image.fromarray(array),
                    target_image=_target_for_record(clean_copy, record),
                    record=record,
                )


def augmented_sid_dataset(
    images_per_label: int,
    seed: int = 4,
    buffer_size: int = 100,
    split: str = "train",
    hf_token: str | None = None,
    *,
    augment: bool = True,
    num_augmentations: int | tuple[int, int] = 6,
    variants_per_image: int = 1,
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
    parallel, and can cache the raw source bytes to local disk.

    ``num_augmentations`` controls how many of the six transforms are chained
    onto *one* output image; ``variants_per_image`` controls how many differently
    -augmented output images each source image produces (dataset expansion) --
    so ``images_per_label=1000, variants_per_image=3`` yields ~9000 training
    samples from ~3000 downloads. The source bytes are still cached/downloaded
    once; the extra variants are generated on read.

    ::

        train = augmented_sid_dataset(1000, split="train", output_size=(224, 224),
                                      num_augmentations=(2, 5), variants_per_image=3,
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
        variants_per_image=variants_per_image,
        output_size=output_size,
        backend=backend,
        num_workers=num_workers,
        prefetch=prefetch,
        cache_dir=cache_dir,
        progress=progress,
    )


def autoencoder_dataset(
    images_per_label: int,
    seed: int = 4,
    buffer_size: int = 100,
    split: str = "train",
    hf_token: str | None = None,
    *,
    augment: bool = True,
    num_augmentations: int | tuple[int, int] = 6,
    variants_per_image: int = 1,
    output_size: tuple[int, int] | None = None,
    backend: str = "thread",
    num_workers: int | None = None,
    prefetch: int | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    progress: bool = True,
) -> AutoencoderDataset:
    """Return a streaming `AutoencoderDataset` of clean/augmented image pairs."""
    return AutoencoderDataset(
        images_per_label,
        seed=seed,
        buffer_size=buffer_size,
        split=split,
        hf_token=hf_token,
        augment=augment,
        num_augmentations=num_augmentations,
        variants_per_image=variants_per_image,
        output_size=output_size,
        backend=backend,
        num_workers=num_workers,
        prefetch=prefetch,
        cache_dir=cache_dir,
        progress=progress,
    )
