from __future__ import annotations

import json
from pathlib import Path

from shared_types.dataset import DatasetSample

_MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "web" / "assets" / "sample_dataset" / "manifest.json"
)
_IMAGES_DIR = _MANIFEST_PATH.parent / "images"


def _load_manifest() -> list[DatasetSample]:
    with open(_MANIFEST_PATH, encoding="utf-8") as f:
        raw_entries = json.load(f)
    return [DatasetSample(**entry) for entry in raw_entries]


def test_manifest_parses_into_dataset_samples():
    samples = _load_manifest()
    assert len(samples) > 0


def test_every_sample_image_file_exists():
    for sample in _load_manifest():
        assert (_IMAGES_DIR / sample.file_name).exists(), sample.file_name


def test_binary_label_matches_label_name():
    for sample in _load_manifest():
        expected = 0 if sample.label_name == "real" else 1
        assert sample.binary_aigc_label == expected


def test_img_ids_are_unique():
    samples = _load_manifest()
    img_ids = [s.img_id for s in samples]
    assert len(img_ids) == len(set(img_ids))
