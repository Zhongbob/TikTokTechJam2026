"""Command-line entry point for building a SID-Set autoencoder subset."""

from __future__ import annotations

import argparse
from pathlib import Path

from packages.image_augmentation.augmenter import ImageAugmenter
from packages.image_augmentation.dataset_builder import AutoencoderDatasetBuilder
from packages.image_augmentation.sid_dataset import load_balanced_sid_subset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream SID-Set and create paired autoencoder data."
    )
    parser.add_argument("--images-per-label", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--split", default="train")
    parser.add_argument("--buffer-size", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    images, metadata = load_balanced_sid_subset(
        images_per_label=args.images_per_label,
        split=args.split,
        seed=args.seed,
        buffer_size=args.buffer_size,
    )

    augmenter = ImageAugmenter(
        output_size=(args.size, args.size),
        seed=args.seed,
    )
    builder = AutoencoderDatasetBuilder(augmenter)
    manifest = builder.build(
        images=images,
        output_dir=args.output_dir,
        source_metadata=metadata,
        overwrite=args.overwrite,
    )

    print(f"Source images: {len(images)}")
    print(f"Training pairs: {len(manifest)}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
