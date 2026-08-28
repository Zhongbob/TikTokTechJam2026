"""Streaming retrieval helpers for the Hugging Face SID-Set dataset."""

from __future__ import annotations

from collections import Counter
from typing import Any

from PIL import Image

from libs.shared_types.augmentation import SourceMetadata

LABEL_NAMES = {
    0: "real",
    1: "synthetic",
    2: "tampered",
}


def load_balanced_sid_subset(
    images_per_label: int,
    split: str = "train",
    seed: int = 4,
    buffer_size: int = 100,
) -> tuple[list[Image.Image], list[SourceMetadata]]:
    """Stream a balanced SID-Set subset without saving source-image copies."""
    if images_per_label <= 0:
        raise ValueError("images_per_label must be greater than zero")
    if buffer_size <= 0:
        raise ValueError("buffer_size must be greater than zero")

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise ImportError(
            "Install Hugging Face Datasets with: pip install datasets"
        ) from error

    stream: Any = load_dataset(
        "saberzl/SID_Set",
        split=split,
        streaming=True,
    )
    stream = stream.shuffle(seed=seed, buffer_size=buffer_size)

    images: list[Image.Image] = []
    metadata: list[SourceMetadata] = []
    counts: Counter[int] = Counter()

    for example in stream:
        label = int(example["label"])
        if label not in LABEL_NAMES or counts[label] >= images_per_label:
            continue

        images.append(example["image"].convert("RGB"))
        metadata.append(
            {
                "img_id": str(example["img_id"]),
                "sid_label": label,
                "label_name": LABEL_NAMES[label],
                "binary_aigc_label": int(label != 0),
            }
        )
        counts[label] += 1

        if all(counts[label] >= images_per_label for label in LABEL_NAMES):
            break

    missing = {
        label: images_per_label - counts[label]
        for label in LABEL_NAMES
        if counts[label] < images_per_label
    }
    if missing:
        raise RuntimeError(f"Stream ended before the balanced subset was filled: {missing}")

    return images, metadata
