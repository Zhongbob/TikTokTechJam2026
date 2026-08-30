"""Autoencoder clean/augmented pair dataset generator."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from shared_types import AugmentationRecord, ImageInput, ImagePairSample, SourceMetadata

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
                source_meta = source_metadata[index] if source_metadata is not None else None
                manifest.extend(self._emit_pairs(
                    index, image, augmented_array, record, inputs_dir, targets_dir, source_meta
                ))
            print(f"Processed {batch_end}/{len(images)} source images")

        return self._write_manifest(manifest, output_dir, start)

    def build_stream(
        self,
        images: Iterable[ImageInput],
        output_dir: str | Path,
        source_metadata: Iterable[SourceMetadata] | None = None,
        num_augmentations: int = 6,
        progress_every: int = 32,
    ) -> list[dict]:
        """Streaming counterpart to :meth:`build`.

        Consumes ``images`` (and ``source_metadata``) lazily, one item at a time,
        so a generator like ``data.dataset_builder.iter_sid_subset()`` can feed a
        dataset of any size without ever materialising it. Each clean/augmented
        pair is written to disk as it is produced; only the manifest (small
        dicts) accumulates in memory.

        Trades away :meth:`build`'s thread/process parallelism for bounded
        memory — use :meth:`build` when the whole input set already fits in RAM
        and you want the CPU throughput.
        """
        output_dir = Path(output_dir)
        inputs_dir, targets_dir = output_dir / "inputs", output_dir / "targets"
        inputs_dir.mkdir(parents=True, exist_ok=True)
        targets_dir.mkdir(parents=True, exist_ok=True)
        manifest, start = [], time.perf_counter()

        metadata_iter = iter(source_metadata) if source_metadata is not None else None
        for index, image in enumerate(images):
            source_meta = next(metadata_iter) if metadata_iter is not None else None
            augmented_array, record = self.augmenter.transform_one(image, num_augmentations)
            manifest.extend(self._emit_pairs(
                index, image, augmented_array, record, inputs_dir, targets_dir, source_meta
            ))
            if progress_every and (index + 1) % progress_every == 0:
                print(f"Processed {index + 1} source images")

        return self._write_manifest(manifest, output_dir, start)

    def _emit_pairs(
        self,
        index: int,
        image: ImageInput,
        augmented_array: np.ndarray,
        record: AugmentationRecord,
        inputs_dir: Path,
        targets_dir: Path,
        source_meta: dict | None,
    ) -> list[dict]:
        """Write the clean and augmented pair for one source image, returning
        their manifest entries. Shared by :meth:`build` and :meth:`build_stream`."""
        clean, source = self.augmenter.load_rgb(image)
        clean_array = np.asarray(clean, dtype=np.uint8)

        # The augmented target should contain the same crop as the input, but
        # none of the other corruptions.
        augmented_target = clean_array
        for step in record.parameters.get("steps", []):
            if step["transform"] == "center_crop":
                crop_ratio = step["parameters"]["crop_ratio"]
                cropped_clean, _ = self.augmenter.center_crop(clean, crop_ratio=crop_ratio)
                augmented_target = np.asarray(cropped_clean, dtype=np.uint8)
                break

        pairs = (
            ("clean", clean_array, clean_array, AugmentationRecord(source, "identity", {})),
            ("augmented", augmented_array, augmented_target, record),
        )

        entries = []
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
            if source_meta is not None:
                entry["source_metadata"] = dict(source_meta)
            entries.append(entry)
        return entries

    @staticmethod
    def _write_manifest(manifest: list[dict], output_dir: Path, start: float) -> list[dict]:
        with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)
        print(f"Created {len(manifest)} pairs in {time.perf_counter() - start:.2f}s at {output_dir}")
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
