"""Real, working implementations of the pipeline's image transforms.

These are standard, deterministic image-processing operations (PIL/numpy) —
unlike the autoencoder/ensemble stages, they are not placeholders.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from shared_types.transforms import (
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


def apply_jpeg_compression(image: Image.Image, quality: int) -> Image.Image:
    rgb_image = image.convert("RGB")
    buffer = io.BytesIO()
    rgb_image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert(image.mode if image.mode in ("RGB", "L") else "RGB")


def apply_gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_resize(image: Image.Image, scale: float) -> Image.Image:
    original_size = image.size
    small_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    downscaled = image.resize(small_size, Image.Resampling.BICUBIC)
    return downscaled.resize(original_size, Image.Resampling.BICUBIC)


def apply_gaussian_noise(
    image: Image.Image, sigma: float, rng: np.random.Generator | None = None
) -> Image.Image:
    rng = rng or np.random.default_rng()
    rgb_image = image.convert("RGB")
    array = np.asarray(rgb_image).astype(np.float32) / 255.0
    noise = rng.normal(0.0, sigma, size=array.shape).astype(np.float32)
    noisy = np.clip(array + noise, 0.0, 1.0)
    return Image.fromarray((noisy * 255).astype(np.uint8), mode="RGB")


def apply_color_jitter(image: Image.Image, factor: float) -> Image.Image:
    result = ImageEnhance.Brightness(image).enhance(factor)
    result = ImageEnhance.Contrast(result).enhance(factor)
    result = ImageEnhance.Color(result).enhance(factor)
    return result


def apply_center_crop(image: Image.Image, crop_fraction: float) -> Image.Image:
    original_size = image.size
    crop_width = round(image.width * crop_fraction)
    crop_height = round(image.height * crop_fraction)
    left = (image.width - crop_width) // 2
    top = (image.height - crop_height) // 2
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize(original_size, Image.Resampling.BICUBIC)


def apply_transform(image: Image.Image, spec: TransformSpec) -> Image.Image:
    """Dispatch to the concrete transform implementation for spec.transform_type."""
    match spec.transform_type:
        case TransformType.JPEG_COMPRESSION:
            assert isinstance(spec.params, JpegCompressionParams)
            return apply_jpeg_compression(image, spec.params.quality)
        case TransformType.GAUSSIAN_BLUR:
            assert isinstance(spec.params, GaussianBlurParams)
            return apply_gaussian_blur(image, spec.params.sigma)
        case TransformType.RESIZE:
            assert isinstance(spec.params, ResizeParams)
            return apply_resize(image, spec.params.scale)
        case TransformType.GAUSSIAN_NOISE:
            assert isinstance(spec.params, GaussianNoiseParams)
            return apply_gaussian_noise(image, spec.params.sigma)
        case TransformType.COLOR_JITTER:
            assert isinstance(spec.params, ColorJitterParams)
            return apply_color_jitter(image, spec.params.factor)
        case TransformType.CENTER_CROP:
            assert isinstance(spec.params, CenterCropParams)
            return apply_center_crop(image, spec.params.crop_fraction)
    raise ValueError(f"Unhandled transform type: {spec.transform_type}")


def apply_transform_pipeline(image: Image.Image, pipeline: TransformPipeline) -> Image.Image:
    """Applies an ordered chain of transform steps, left to right.

    An empty pipeline returns the image unchanged.
    """
    result = image
    for spec in pipeline:
        result = apply_transform(result, spec)
    return result
