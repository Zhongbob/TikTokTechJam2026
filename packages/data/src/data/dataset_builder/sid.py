"""Stream balanced subsets from saberzl/SID_Set without saving source copies."""

from __future__ import annotations

from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

from PIL import Image
from shared_types import LabeledImageSample, SourceMetadata

LABEL_NAMES = {0: "real", 1: "synthetic", 2: "tampered"}


def _iter_sid_raw(
    images_per_label: int, seed: int, buffer_size: int, split: str, hf_token: str | None, decode: bool
) -> Iterator[tuple[Any, SourceMetadata]]:
    """Shared balanced-subset streaming core.

    Yields ``(payload, metadata)`` where ``payload`` is a decoded PIL image
    when ``decode`` is True, otherwise the raw ``{"bytes", "path"}`` dict from
    the Hugging Face ``Image`` feature. Stops as soon as every label's quota
    is met; raises if the stream runs out first.
    """
    # Lazy import keeps local-folder generation usable without Hugging Face.
    import os
    if hf_token is not None:
        os.environ["HF_TOKEN"] = hf_token

    from datasets import load_dataset

    if images_per_label < 1:
        raise ValueError("images_per_label must be at least 1")
    dataset = load_dataset("saberzl/SID_Set", split=split, streaming=True)
    if not decode:
        from datasets import Image as _HFImage

        dataset = dataset.cast_column("image", _HFImage(decode=False))
    stream = dataset.shuffle(seed=seed, buffer_size=buffer_size)

    counts: Counter[int] = Counter()
    for example in stream:
        label = int(example["label"])
        if label not in LABEL_NAMES or counts[label] >= images_per_label:
            continue
        metadata: SourceMetadata = {
            "img_id": str(example["img_id"]),
            "sid_label": label,
            "label_name": LABEL_NAMES[label],
            "binary_aigc_label": int(label != 0),
        }
        counts[label] += 1
        yield example["image"], metadata
        if all(counts[label] >= images_per_label for label in LABEL_NAMES):
            break
    if not all(counts[label] >= images_per_label for label in LABEL_NAMES):
        raise RuntimeError(f"Could not retrieve requested balanced subset; counts={dict(counts)}")


def iter_sid_subset(
    images_per_label: int, seed: int = 4, buffer_size: int = 100, split: str = "train", hf_token: str = None
) -> Iterator[LabeledImageSample]:
    """Yield `LabeledImageSample`s one at a time from a balanced SID-Set subset.

    Unlike :func:`load_sid_subset`, nothing is accumulated in memory: each PIL image
    is produced lazily and can be discarded by the caller before the next is fetched,
    keeping peak memory to a single decoded image (plus the shuffle buffer). The
    result is a single-use generator; see :func:`sid_subset_factory` when the same
    subset must be iterated more than once, or :func:`iter_sid_encoded` to move
    image decoding off the streaming thread.
    """
    for image, metadata in _iter_sid_raw(images_per_label, seed, buffer_size, split, hf_token, decode=True):
        yield LabeledImageSample(image=image.convert("RGB"), metadata=metadata)


def iter_sid_encoded(
    images_per_label: int, seed: int = 4, buffer_size: int = 100, split: str = "train", hf_token: str = None
) -> Iterator[tuple[bytes, SourceMetadata]]:
    """Like :func:`iter_sid_subset` but yields ``(raw encoded image bytes, metadata)``.

    The streaming thread never pays the PNG/JPEG decode -- the caller can push
    that onto a worker pool (`ImageAugmenter` accepts ``bytes`` directly), which
    is the main lever when a plain `iter_sid_subset()` is decode-bound.
    """
    for entry, metadata in _iter_sid_raw(images_per_label, seed, buffer_size, split, hf_token, decode=False):
        data = entry["bytes"]
        if data is None:  # feature backed by a local file rather than inline bytes
            data = Path(entry["path"]).read_bytes()
        yield data, metadata


def load_sid_subset(images_per_label: int, seed: int = 4, buffer_size: int = 100, split: str = "train", hf_token: str = None):
    """Materializes `iter_sid_subset()` into (images, metadata) lists.

    Convenient for small pulls you'll reuse more than once (e.g. a
    validation set used across train/evaluate/inference cells), but holds
    every selected image in memory at once — prefer `iter_sid_subset()`
    directly for a large, single-use training pull.
    """
    images: list[Image.Image] = []
    metadata: list[SourceMetadata] = []
    for sample in iter_sid_subset(images_per_label, seed=seed, buffer_size=buffer_size, split=split, hf_token=hf_token):
        images.append(sample.image)
        metadata.append(sample.metadata)
    return images, metadata


def sid_subset_factory(
    images_per_label: int, seed: int = 4, buffer_size: int = 100, split: str = "train", hf_token: str = None
) -> Callable[[], Iterator[LabeledImageSample]]:
    """Return a zero-arg callable that yields a FRESH `iter_sid_subset()` stream each call.

    A generator is single-use, but a validation set is normally iterated more
    than once (the trainer's own val pass, `evaluate()`, an inference demo).
    Materializing it with `load_sid_subset()` + `to_labeled_samples()` holds
    every image in RAM for the whole session; calling this factory once and
    invoking the result at each use site re-streams the subset instead, so only
    one decoded image is ever live. The pull is deterministic (fixed `seed`), so
    every call sees the same images in the same order.

        make_val = sid_subset_factory(images_per_label=200, split="validation")
        trainer.train(train_stream, val_samples=make_val())
        trainer.evaluate(make_val())
    """
    return partial(
        iter_sid_subset,
        images_per_label,
        seed=seed,
        buffer_size=buffer_size,
        split=split,
        hf_token=hf_token,
    )


def to_labeled_samples(
    images: Sequence[Image.Image], metadata: Sequence[SourceMetadata]
) -> list[LabeledImageSample]:
    """Zips `load_sid_subset()`'s (images, metadata) output into the shared
    `LabeledImageSample` type that `TrainableModel.train()` implementations
    (e.g. our_classifier, ensemble) expect.

        images, metadata = load_sid_subset(images_per_label=200)
        samples = to_labeled_samples(images, metadata)
        trainer.train(samples)
    """
    if len(images) != len(metadata):
        raise ValueError(f"images ({len(images)}) and metadata ({len(metadata)}) must be the same length")
    return [LabeledImageSample(image=image, metadata=meta) for image, meta in zip(images, metadata)]
