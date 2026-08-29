"""Autoencoder clean/augmented pair dataset generator."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from shared_types import AugmentationRecord, ImageInput

from data.augmentation import ImageAugmenter


class AutoencoderDatasetBuilder:
    """Create N clean→clean and N augmented→clean pairs (2N total)."""

    def __init__(self, augmenter: ImageAugmenter) -> None:
        self.augmenter = augmenter

    def build(
        self,
        images: Sequence[ImageInput],
        output_dir: str | Path,
        source_metadata: Sequence[dict] | None = None,
        num_augmentations: int = 6,
        backend: str = "thread",
        num_workers: int | None = None,
        batch_size: int = 32,
    ) -> list[dict]:
        if source_metadata is not None and len(source_metadata) != len(images):
            raise ValueError("source_metadata must have one entry per image")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        output_dir = Path(output_dir)
        inputs_dir, targets_dir = output_dir / "inputs", output_dir / "targets"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        targets_dir.mkdir(parents=True, exist_ok=True)
        manifest, start = [], time.perf_counter()

        for batch_start in range(0, len(images), batch_size):
            batch_end = min(batch_start + batch_size, len(images))
            batch_images = images[batch_start:batch_end]
            augmented, records = self.augmenter.transform_images(
                batch_images, num_augmentations, True, backend, num_workers
            )
            for offset, (image, augmented_array, record) in enumerate(zip(batch_images, list(augmented), records)):
                index = batch_start + offset
                clean, source = self.augmenter.load_rgb(image)
                clean_array = np.asarray(clean, dtype=np.uint8)
                pairs = (
                    ("clean", clean_array, AugmentationRecord(source, "identity", {})),
                    ("augmented", augmented_array, record),
                )
                for variant, input_array, applied in pairs:
                    filename = f"{index:06d}_{variant}.png"
                    input_path, target_path = inputs_dir / filename, targets_dir / filename
                    Image.fromarray(input_array).save(input_path)
                    Image.fromarray(clean_array).save(target_path)
                    entry = {
                        "source_index": index, "source": source, "variant": variant,
                        "input_path": str(input_path), "target_path": str(target_path),
                        "transform": applied.transform, "parameters": applied.parameters,
                    }
                    if source_metadata is not None:
                        entry["source_metadata"] = dict(source_metadata[index])
                    manifest.append(entry)
            print(f"Processed {batch_end}/{len(images)} source images")

        with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)
        print(f"Created {len(manifest)} pairs in {time.perf_counter() - start:.2f}s at {output_dir}")
        return manifest
