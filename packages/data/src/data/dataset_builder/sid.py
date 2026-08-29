"""Stream balanced subsets from saberzl/SID_Set without saving source copies."""

from __future__ import annotations

from collections import Counter
from typing import Iterator, Sequence

from PIL import Image
from shared_types import LabeledImageSample, SourceMetadata

LABEL_NAMES = {0: "real", 1: "synthetic", 2: "tampered"}


def load_sid_subset(images_per_label: int, seed: int = 4, buffer_size: int = 100, hf_token: str = None):
    # Lazy import keeps local-folder generation usable without Hugging Face.
    import os
    if hf_token is not None:
        os.environ["HF_TOKEN"] = hf_token

    from datasets import load_dataset

    if images_per_label < 1:
        raise ValueError("images_per_label must be at least 1")
    stream = load_dataset("saberzl/SID_Set", split=split, streaming=True).shuffle(
        seed=seed, buffer_size=buffer_size
    )
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
        yield LabeledImageSample(image=example["image"].convert("RGB"), metadata=metadata)
        if all(counts[label] >= images_per_label for label in LABEL_NAMES):
            break
    if not all(counts[label] >= images_per_label for label in LABEL_NAMES):
        raise RuntimeError(f"Could not retrieve requested balanced subset; counts={dict(counts)}")


def load_sid_subset(images_per_label: int, seed: int = 4, buffer_size: int = 100, split: str = "train"):
    """Materializes `iter_sid_subset()` into (images, metadata) lists.

    Convenient for small pulls you'll reuse more than once (e.g. a
    validation set used across train/evaluate/inference cells), but holds
    every selected image in memory at once — prefer `iter_sid_subset()`
    directly for a large, single-use training pull.
    """
    images: list[Image.Image] = []
    metadata: list[SourceMetadata] = []
    for sample in iter_sid_subset(images_per_label, seed=seed, buffer_size=buffer_size, split=split):
        images.append(sample.image)
        metadata.append(sample.metadata)
    return images, metadata


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
