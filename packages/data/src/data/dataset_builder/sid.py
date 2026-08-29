"""Stream balanced subsets from saberzl/SID_Set without saving source copies."""

from __future__ import annotations

from collections import Counter
from typing import Iterator

LABEL_NAMES = {0: "real", 1: "synthetic", 2: "tampered"}


def iter_sid_subset(
    images_per_label: int, seed: int = 4, buffer_size: int = 100, hf_token: str = None
) -> Iterator[tuple["object", dict]]:
    """Yield ``(image, metadata)`` pairs one at a time from a balanced SID-Set subset.

    Unlike :func:`load_sid_subset`, nothing is accumulated in memory: each PIL image
    is produced lazily and can be discarded by the caller before the next is fetched,
    keeping peak memory to a single decoded image (plus the shuffle buffer).
    """
    # Lazy import keeps local-folder generation usable without Hugging Face.
    import os
    if hf_token is not None:
        os.environ["HF_TOKEN"] = hf_token

    from datasets import load_dataset

    if images_per_label < 1:
        raise ValueError("images_per_label must be at least 1")
    stream = load_dataset("saberzl/SID_Set", split="train", streaming=True).shuffle(
        seed=seed, buffer_size=buffer_size
    )
    counts = Counter()
    for example in stream:
        label = int(example["label"])
        if label not in LABEL_NAMES or counts[label] >= images_per_label:
            continue
        metadata = {
            "img_id": str(example["img_id"]), "sid_label": label,
            "label_name": LABEL_NAMES[label], "binary_aigc_label": int(label != 0),
        }
        yield example["image"].convert("RGB"), metadata
        counts[label] += 1
        if all(counts[label] >= images_per_label for label in LABEL_NAMES):
            break
    if not all(counts[label] >= images_per_label for label in LABEL_NAMES):
        raise RuntimeError(f"Could not retrieve requested balanced subset; counts={dict(counts)}")


def load_sid_subset(images_per_label: int, seed: int = 4, buffer_size: int = 100, hf_token: str = None):
    images, metadata = [], []
    for image, entry in iter_sid_subset(images_per_label, seed, buffer_size, hf_token):
        images.append(image)
        metadata.append(entry)
    return images, metadata
