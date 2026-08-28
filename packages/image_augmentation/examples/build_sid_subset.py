"""Example: stream a balanced SID-Set subset and build paired data."""

from packages.image_augmentation import (
    AutoencoderDatasetBuilder,
    ImageAugmenter,
    load_balanced_sid_subset,
)


def main() -> None:
    images, metadata = load_balanced_sid_subset(
        images_per_label=2,
        seed=4,
    )
    augmenter = ImageAugmenter(output_size=(256, 256), seed=4)
    builder = AutoencoderDatasetBuilder(augmenter)
    manifest = builder.build(
        images=images,
        output_dir="outputs/sid_test",
        source_metadata=metadata,
    )
    print(f"Created {len(manifest)} pairs")


if __name__ == "__main__":
    main()
