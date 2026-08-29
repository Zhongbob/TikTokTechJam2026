"""Command-line entry point for autoencoder dataset generation."""

from __future__ import annotations

import argparse
import os

from image_io import find_images

from data.augmentation import ImageAugmenter
from data.dataset_builder import AutoencoderDatasetBuilder, load_sid_subset


def parse_size(value: str) -> tuple[int, int]:
    try:
        width, height = value.lower().split("x")
        return int(width), int(height)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use WIDTHxHEIGHT, for example 256x256") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a 2x clean/augmented autoencoder dataset.")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--num-augmentations", type=int, default=6, choices=range(1, 7))
    parser.add_argument("--backend", choices=("sequential", "thread", "process"), default="process")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 2))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-size", type=parse_size, default=(256, 256), metavar="WIDTHxHEIGHT")
    parser.add_argument("--seed", type=int, default=4)

    subparsers = parser.add_subparsers(dest="source", required=True)
    local = subparsers.add_parser("local", help="Read PNG/JPEG files from a folder")
    local.add_argument("--input", required=True, help="Input image directory")
    local.add_argument("--no-recursive", action="store_true")
    sid = subparsers.add_parser("sid", help="Stream a balanced SID-Set subset")
    sid.add_argument("--images-per-label", type=int, default=2)
    sid.add_argument("--shuffle-buffer", type=int, default=100)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = None
    if args.source == "local":
        images = find_images(args.input, recursive=not args.no_recursive)
    else:
        images, metadata = load_sid_subset(args.images_per_label, args.seed, args.shuffle_buffer)

    augmenter = ImageAugmenter(output_size=args.output_size, seed=args.seed)
    builder = AutoencoderDatasetBuilder(augmenter)
    builder.build(
        images=images, output_dir=args.output, source_metadata=metadata,
        num_augmentations=args.num_augmentations, backend=args.backend,
        num_workers=args.workers, batch_size=args.batch_size,
    )


if __name__ == "__main__":
    # Required for safe multiprocessing, particularly on Windows/macOS.
    main()
