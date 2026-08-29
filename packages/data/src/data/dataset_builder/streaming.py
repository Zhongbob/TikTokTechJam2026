"""Re-iterable, memory-bounded SID-Set stream with data-package augmentation.

`iter_sid_subset()` streams raw SID-Set samples; this module wraps it so every
image is run through `data.augmentation.ImageAugmenter` -- the six realistic
corruptions from the problem statement (JPEG compression, Gaussian blur,
resize, Gaussian noise, colour jitter, centre crop) -- before it is yielded.

The result is a drop-in replacement for `to_labeled_samples(load_sid_subset(...))`
wherever a `shared_types.TrainableModel.train()` / `.evaluate()` consumes an
`Iterable[LabeledImageSample]` (e.g. `NormalClassifierTrainer`), except that
the whole subset is never held in memory -- only one decoded image at a time.
"""

from __future__ import annotations

from typing import Iterator

from PIL import Image
from shared_types import LabeledImageSample

from data.augmentation import ImageAugmenter

from .sid import LABEL_NAMES, iter_sid_subset


class AugmentedSIDDataset:
    """A re-iterable stream of (optionally augmented) SID-Set samples.

    Takes the same positional arguments as `load_sid_subset()`, plus
    augmentation controls. Every `iter(dataset)` opens a *fresh*
    `iter_sid_subset()` stream, so iterating twice -- a second training
    epoch, or `evaluate()` after `train()` -- re-streams from Hugging Face
    rather than replaying a cached list. The pull is seed-deterministic, so
    every pass sees the same source images in the same order (and, because
    the augmenter is re-seeded from `seed` each pass, the same corruptions).

    Peak memory is one decoded image plus PIL's working buffers -- the
    subset itself is never materialised.

    Args:
        images_per_label: balanced count per SID label (real / synthetic /
            tampered), so the stream yields ``3 * images_per_label`` samples.
        seed, buffer_size, split, hf_token: forwarded to `iter_sid_subset()`.
        augment: when False, images are passed through untouched (only
            resized if ``output_size`` is set) -- use this for a validation
            or test stream you want left clean.
        num_augmentations: how many of the six transforms to apply per image
            (1-6), forwarded to `ImageAugmenter.transform_one()`.
        output_size: (width, height) to resize every image to before
            augmenting; None keeps native resolution.
    """

    def __init__(
        self,
        images_per_label: int,
        seed: int = 4,
        buffer_size: int = 100,
        split: str = "train",
        hf_token: str | None = None,
        *,
        augment: bool = True,
        num_augmentations: int = 6,
        output_size: tuple[int, int] | None = None,
    ) -> None:
        if images_per_label < 1:
            raise ValueError("images_per_label must be at least 1")
        self.images_per_label = images_per_label
        self.seed = seed
        self.buffer_size = buffer_size
        self.split = split
        self.hf_token = hf_token
        self.augment = augment
        self.num_augmentations = num_augmentations
        self.output_size = output_size

    def __len__(self) -> int:
        """Known up front: the stream is balanced with `images_per_label`
        samples per SID label. Handy for progress bars; if the underlying
        stream can't fill the subset, iteration raises rather than returning
        fewer than this."""
        return self.images_per_label * len(LABEL_NAMES)

    def __repr__(self) -> str:
        mode = f"augment x{self.num_augmentations}" if self.augment else "clean"
        return (
            f"AugmentedSIDDataset(split={self.split!r}, per_label={self.images_per_label}, "
            f"{mode}, output_size={self.output_size})"
        )

    def __iter__(self) -> Iterator[LabeledImageSample]:
        needs_augmenter = self.augment or self.output_size is not None
        augmenter = ImageAugmenter(output_size=self.output_size, seed=self.seed) if needs_augmenter else None

        for sample in iter_sid_subset(
            self.images_per_label,
            seed=self.seed,
            buffer_size=self.buffer_size,
            split=self.split,
            hf_token=self.hf_token,
        ):
            if augmenter is None:
                yield sample
                continue
            if self.augment:
                array, _record = augmenter.transform_one(sample.image, self.num_augmentations)
                image = Image.fromarray(array)
            else:  # resize only
                image, _ = augmenter.load_rgb(sample.image)
            yield LabeledImageSample(image=image, metadata=sample.metadata)


def augmented_sid_dataset(
    images_per_label: int,
    seed: int = 4,
    buffer_size: int = 100,
    split: str = "train",
    hf_token: str | None = None,
    *,
    augment: bool = True,
    num_augmentations: int = 6,
    output_size: tuple[int, int] | None = None,
) -> AugmentedSIDDataset:
    """Return a re-iterable `AugmentedSIDDataset` (see that class for details).

    Same positional signature as `load_sid_subset()`; the return value is a
    streaming, memory-bounded stand-in for
    `to_labeled_samples(load_sid_subset(...))` that also applies the data
    package's `ImageAugmenter` transforms. Feed it straight to
    `NormalClassifierTrainer.train()` / `.evaluate()` or any other
    `TrainableModel` that consumes `Iterable[LabeledImageSample]`::

        train = augmented_sid_dataset(1000, split="train", output_size=(224, 224))
        val = augmented_sid_dataset(200, split="validation", augment=False, output_size=(224, 224))
        trainer.train(train, val_samples=val)
        trainer.evaluate(val)          # re-iterates -> re-streams
    """
    return AugmentedSIDDataset(
        images_per_label,
        seed=seed,
        buffer_size=buffer_size,
        split=split,
        hf_token=hf_token,
        augment=augment,
        num_augmentations=num_augmentations,
        output_size=output_size,
    )
