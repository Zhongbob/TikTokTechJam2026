from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from shared_types.transforms import (
    CenterCropParams,
    ColorJitterParams,
    GaussianBlurParams,
    GaussianNoiseParams,
    JpegCompressionParams,
    ResizeParams,
    TransformSpec,
    TransformType,
)

from web.services.transforms import (
    apply_center_crop,
    apply_color_jitter,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_jpeg_compression,
    apply_resize,
    apply_transform,
    apply_transform_pipeline,
)


@pytest.fixture
def sample_image() -> Image.Image:
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    return Image.fromarray(array, mode="RGB")


def test_jpeg_compression_preserves_size_and_changes_pixels(sample_image):
    result = apply_jpeg_compression(sample_image, quality=30)
    assert result.size == sample_image.size
    assert list(result.getdata()) != list(sample_image.getdata())


def test_gaussian_blur_preserves_size_and_changes_pixels(sample_image):
    result = apply_gaussian_blur(sample_image, sigma=2.0)
    assert result.size == sample_image.size
    assert list(result.getdata()) != list(sample_image.getdata())


def test_resize_round_trips_to_original_size(sample_image):
    result = apply_resize(sample_image, scale=0.25)
    assert result.size == sample_image.size


def test_gaussian_noise_is_reproducible_with_seeded_generator(sample_image):
    result_a = apply_gaussian_noise(sample_image, sigma=0.05, rng=np.random.default_rng(42))
    result_b = apply_gaussian_noise(sample_image, sigma=0.05, rng=np.random.default_rng(42))
    assert list(result_a.getdata()) == list(result_b.getdata())
    assert list(result_a.getdata()) != list(sample_image.getdata())


def test_color_jitter_preserves_size_and_changes_pixels(sample_image):
    result = apply_color_jitter(sample_image, factor=1.2)
    assert result.size == sample_image.size
    assert list(result.getdata()) != list(sample_image.getdata())


def test_center_crop_round_trips_to_original_size(sample_image):
    result = apply_center_crop(sample_image, crop_fraction=0.8)
    assert result.size == sample_image.size


def test_apply_transform_dispatches_correctly(sample_image):
    spec = TransformSpec(TransformType.JPEG_COMPRESSION, JpegCompressionParams(quality=50))
    result = apply_transform(sample_image, spec)
    assert result.size == sample_image.size


def test_apply_transform_pipeline_applies_steps_in_order(sample_image):
    pipeline = [
        TransformSpec(TransformType.GAUSSIAN_BLUR, GaussianBlurParams(sigma=1.0)),
        TransformSpec(TransformType.COLOR_JITTER, ColorJitterParams(factor=1.2)),
        TransformSpec(TransformType.CENTER_CROP, CenterCropParams(crop_fraction=0.8)),
    ]
    result = apply_transform_pipeline(sample_image, pipeline)
    assert result.size == sample_image.size
    assert list(result.getdata()) != list(sample_image.getdata())

    # sequential application, not just the last step applied to the original
    step_by_step = sample_image
    for spec in pipeline:
        step_by_step = apply_transform(step_by_step, spec)
    assert list(result.getdata()) == list(step_by_step.getdata())


def test_apply_transform_pipeline_empty_is_identity(sample_image):
    result = apply_transform_pipeline(sample_image, [])
    assert list(result.getdata()) == list(sample_image.getdata())
