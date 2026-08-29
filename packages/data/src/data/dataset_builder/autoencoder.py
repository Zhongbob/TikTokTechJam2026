"""Autoencoder clean/augmented pair dataset generator."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from shared_types import AugmentationRecord, ImageInput, ImagePairSample

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

                # Editing this part to basically find the crop value and return the cropped clean image as final target
                # The augmented target should contain the same crop as the input,
                # but none of the other corruptions.
                augmented_target = clean_array

                for step in record.parameters.get("steps", []):
                    if step["transform"] == "center_crop":
                        crop_ratio = step["parameters"]["crop_ratio"]
                        cropped_clean, _ = self.augmenter.center_crop(
                            clean,
                            crop_ratio=crop_ratio,
                        )
                        augmented_target = np.asarray(
                            cropped_clean, dtype=np.uint8)
                        break

                pairs = (
                    (
                        "clean",
                        clean_array,
                        clean_array,
                        AugmentationRecord(source, "identity", {}),
                    ),
                    (
                        "augmented",
                        augmented_array,
                        augmented_target,
                        record,
                    ),
                )

                for variant, input_array, target_array, applied in pairs:
                    filename = f"{index:06d}_{variant}.png"
                    input_path, target_path = inputs_dir / filename, targets_dir / filename
                    Image.fromarray(input_array).save(input_path)
                    Image.fromarray(target_array).save(target_path)
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
        print(
            f"Created {len(manifest)} pairs in {time.perf_counter() - start:.2f}s at {output_dir}")
        return manifest


def load_manifest_as_samples(output_dir: str | Path) -> list[ImagePairSample]:
    """Loads a directory built by `AutoencoderDatasetBuilder.build()` into the
    shared `ImagePairSample` type that `TrainableModel.train()`
    implementations (e.g. the autoencoder) expect.

        builder.build(images, "outputs/local")
        samples = load_manifest_as_samples("outputs/local")
        trainer.train(samples)
    """
    output_dir = Path(output_dir)
    with (output_dir / "manifest.json").open(encoding="utf-8") as file:
        manifest = json.load(file)

    samples = []
    for entry in manifest:
        with Image.open(entry["input_path"]) as opened:
            input_image = opened.convert("RGB").copy()
        with Image.open(entry["target_path"]) as opened:
            target_image = opened.convert("RGB").copy()
        record = AugmentationRecord(
            source=entry["source"], transform=entry["transform"], parameters=entry["parameters"]
        )
        samples.append(ImagePairSample(input_image=input_image, target_image=target_image, record=record))
    return samples
