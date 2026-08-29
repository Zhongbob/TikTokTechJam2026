"""Stream balanced subsets from saberzl/SID_Set without saving source copies."""

from __future__ import annotations

from collections import Counter

LABEL_NAMES = {0: "real", 1: "synthetic", 2: "tampered"}


def load_sid_subset(images_per_label: int, seed: int = 4, buffer_size: int = 100):
    # Lazy import keeps local-folder generation usable without Hugging Face.
    from datasets import load_dataset

    if images_per_label < 1:
        raise ValueError("images_per_label must be at least 1")
    stream = load_dataset("saberzl/SID_Set", split="train", streaming=True).shuffle(
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
