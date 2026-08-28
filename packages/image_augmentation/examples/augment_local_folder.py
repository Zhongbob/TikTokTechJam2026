"""Example: augment all PNG/JPEG images in a local directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from packages.image_augmentation import ImageAugmenter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    iterator = args.input_dir.rglob("*") if args.recursive else args.input_dir.iterdir()
    paths = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not paths:
        raise SystemExit(f"No PNG/JPEG images found in {args.input_dir}")

    output_size = (args.size, args.size) if args.size else None
    augmenter = ImageAugmenter(output_size=output_size, seed=args.seed)
    _, saved_paths, records = augmenter.transform_and_save(
        paths,
        output_dir=args.output_dir,
    )

    for source, destination, record in zip(paths, saved_paths, records):
        print(f"{source} -> {destination}")
        print("  order:", " -> ".join(record.parameters["order"]))


if __name__ == "__main__":
    main()
