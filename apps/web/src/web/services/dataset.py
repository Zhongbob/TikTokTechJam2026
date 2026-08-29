"""Loading and random-picking for the bundled placeholder sample dataset.

The dataset itself (assets/sample_dataset/) is procedurally generated
placeholder imagery — see scripts/generate_sample_dataset.py. Swap it for a
real curated dataset later by replacing the manifest + images on disk; this
module's interface does not need to change.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import streamlit as st
from PIL import Image
from shared_types.dataset import DatasetSample
from shared_types.transforms import (
    BLUR_SIGMAS,
    CENTER_CROP_FRACTION,
    COLOR_JITTER_RANGE,
    JPEG_QUALITIES,
    NOISE_SIGMAS,
    RESIZE_SCALES,
    CenterCropParams,
    ColorJitterParams,
    GaussianBlurParams,
    GaussianNoiseParams,
    JpegCompressionParams,
    ResizeParams,
    TransformPipeline,
    TransformSpec,
    TransformType,
)

_DATASET_DIR = Path(__file__).resolve().parent.parent / "assets" / "sample_dataset"
_MANIFEST_PATH = _DATASET_DIR / "manifest.json"
_IMAGES_DIR = _DATASET_DIR / "images"


@st.cache_data
def load_sample_manifest() -> list[DatasetSample]:
    with open(_MANIFEST_PATH, encoding="utf-8") as f:
        raw_entries = json.load(f)
    return [DatasetSample(**entry) for entry in raw_entries]


def load_sample_image(sample: DatasetSample) -> Image.Image:
    return Image.open(_IMAGES_DIR / sample.file_name).convert("RGB")


def pick_random_sample(
    manifest: list[DatasetSample], rng: random.Random | None = None
) -> DatasetSample:
    rng = rng or random.Random()
    return rng.choice(manifest)


def _random_transform_spec(transform_type: TransformType, rng: random.Random) -> TransformSpec:
    match transform_type:
        case TransformType.JPEG_COMPRESSION:
            params = JpegCompressionParams(quality=rng.choice(JPEG_QUALITIES))
        case TransformType.GAUSSIAN_BLUR:
            params = GaussianBlurParams(sigma=rng.choice(BLUR_SIGMAS))
        case TransformType.RESIZE:
            params = ResizeParams(scale=rng.choice(RESIZE_SCALES))
        case TransformType.GAUSSIAN_NOISE:
            params = GaussianNoiseParams(sigma=rng.choice(NOISE_SIGMAS))
        case TransformType.COLOR_JITTER:
            params = ColorJitterParams(factor=rng.uniform(*COLOR_JITTER_RANGE))
        case TransformType.CENTER_CROP:
            params = CenterCropParams(crop_fraction=CENTER_CROP_FRACTION)
        case _:
            raise ValueError(f"Unhandled transform type: {transform_type}")

    return TransformSpec(transform_type=transform_type, params=params)


def pick_random_transform_pipeline(rng: random.Random | None = None) -> TransformPipeline:
    """Picks a random chain of 1-3 distinct transform steps, in random order."""
    rng = rng or random.Random()
    step_count = rng.randint(1, 3)
    transform_types = rng.sample(list(TransformType), k=step_count)
    return [_random_transform_spec(t, rng) for t in transform_types]
