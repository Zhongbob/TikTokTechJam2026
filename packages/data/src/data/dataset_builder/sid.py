"""Stream balanced subsets from saberzl/SID_Set without saving source copies."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from PIL import Image
from shared_types import LabeledImageSample, SourceMetadata

LABEL_NAMES = {0: "real", 1: "synthetic", 2: "tampered"}


def load_sid_subset(images_per_label: int, seed: int = 4, buffer_size: int = 100, split: str = "train"):
    # Lazy import keeps local-folder generation usable without Hugging Face.
    from datasets import load_dataset

    if images_per_label < 1:
        raise ValueError("images_per_label must be at least 1")
    stream = load_dataset("saberzl/SID_Set", split=split, streaming=True).shuffle(
        seed=seed, buffer_size=buffer_size
    )
    images, metadata, counts = [], [], Counter()
    for example in stream:
        label = int(example["label"])
        if label not in LABEL_NAMES or counts[label] >= images_per_label:
            continue
        images.append(example["image"].convert("RGB"))
        metadata.append({
            "img_id": str(example["img_id"]), "sid_label": label,
            "label_name": LABEL_NAMES[label], "binary_aigc_label": int(label != 0),
        })
        counts[label] += 1
        if all(counts[label] >= images_per_label for label in LABEL_NAMES):
            break
    if not all(counts[label] >= images_per_label for label in LABEL_NAMES):
        raise RuntimeError(f"Could not retrieve requested balanced subset; counts={dict(counts)}")
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
