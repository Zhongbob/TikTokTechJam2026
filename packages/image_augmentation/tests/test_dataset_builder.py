import json

import numpy as np
from PIL import Image

from packages.image_augmentation import AutoencoderDatasetBuilder, ImageAugmenter


def test_builder_creates_two_pairs_per_image(tmp_path) -> None:
    images = [
        np.full((24, 24, 3), 64, dtype=np.uint8),
        np.full((24, 24, 3), 192, dtype=np.uint8),
    ]
    builder = AutoencoderDatasetBuilder(ImageAugmenter(seed=4))

    manifest = builder.build(images, tmp_path / "dataset")

    assert len(manifest) == 4
    assert (tmp_path / "dataset" / "manifest.json").exists()
    loaded_manifest = json.loads(
        (tmp_path / "dataset" / "manifest.json").read_text()
    )
    assert len(loaded_manifest) == 4

    for record in manifest:
        input_path = tmp_path / "dataset" / "inputs" / (
            f"{record['source_index']:06d}_{record['variant']}.png"
        )
        target_path = tmp_path / "dataset" / "targets" / input_path.name
        assert input_path.exists()
        assert target_path.exists()

    clean_input = np.asarray(
        Image.open(tmp_path / "dataset" / "inputs" / "000000_clean.png")
    )
    clean_target = np.asarray(
        Image.open(tmp_path / "dataset" / "targets" / "000000_clean.png")
    )
    assert np.array_equal(clean_input, clean_target)
