"""Paired autoencoder dataset generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from packages.image_augmentation.augmenter import ImageAugmenter
from libs.shared_types.augmentation import (
    AugmentationRecord,
    ImageInput,
    SourceMetadata,
)


class AutoencoderDatasetBuilder:
    """Write clean→clean and augmented→clean image pairs."""

    def __init__(self, augmenter: ImageAugmenter) -> None:
        self.augmenter = augmenter

    def build(
        self,
        images: Sequence[ImageInput],
        output_dir: str | Path,
        source_metadata: Sequence[SourceMetadata] | None = None,
        overwrite: bool = False,
    ) -> list[dict]:
        """Create a 2N paired dataset and return its manifest.

        Source images supplied as PIL objects remain in memory. Only the paired
        autoencoder inputs and targets are written.
        """
        if source_metadata is not None and len(source_metadata) != len(images):
            raise ValueError("source_metadata must have one entry per image")

        root = Path(output_dir)
        if root.exists() and any(root.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {root}. "
                "Choose another directory or pass overwrite=True."
            )

        inputs_dir = root / "inputs"
        targets_dir = root / "targets"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        targets_dir.mkdir(parents=True, exist_ok=True)

        manifest: list[dict] = []
        for index, image in enumerate(images):
            clean, source = self.augmenter.load_rgb(image)
            clean_array = np.asarray(clean, dtype=np.uint8)
            augmented_array, augmentation_record = self.augmenter.transform_one(image)

            pairs = (
                (
                    "clean",
                    clean_array,
                    AugmentationRecord(source, "identity", {}),
                ),
                ("augmented", augmented_array, augmentation_record),
            )

            for variant, input_array, record in pairs:
                filename = f"{index:06d}_{variant}.png"
                input_path = inputs_dir / filename
                target_path = targets_dir / filename

                Image.fromarray(input_array).save(input_path, format="PNG")
                Image.fromarray(clean_array).save(target_path, format="PNG")

                entry = {
                    "source_index": index,
                    "source": source,
                    "variant": variant,
                    "input_path": str(input_path),
                    "target_path": str(target_path),
                    "transform": record.transform,
                    "parameters": record.parameters,
                }
                if source_metadata is not None:
                    entry["source_metadata"] = dict(source_metadata[index])
                manifest.append(entry)

        manifest_path = root / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)

        return manifest
