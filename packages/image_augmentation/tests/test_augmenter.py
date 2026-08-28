import numpy as np

from packages.image_augmentation import ImageAugmenter


def test_transform_one_applies_all_five_transforms() -> None:
    image = np.full((48, 64, 3), 128, dtype=np.uint8)
    augmenter = ImageAugmenter(output_size=(32, 32), seed=4)

    transformed, record = augmenter.transform_one(image)

    assert transformed.shape == (32, 32, 3)
    assert record.transform == "permutation_all_5"
    assert len(record.parameters["order"]) == 5
    assert set(record.parameters["order"]) == set(augmenter.TRANSFORMS)
    assert len(record.parameters["steps"]) == 5


def test_seed_makes_results_reproducible() -> None:
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    first = ImageAugmenter(seed=7)
    second = ImageAugmenter(seed=7)

    first_image, first_record = first.transform_one(image)
    second_image, second_record = second.transform_one(image)

    assert np.array_equal(first_image, second_image)
    assert first_record.parameters == second_record.parameters
