from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias


class TransformType(str, Enum):
    """Kinds of image transformation the pipeline can apply."""

    JPEG_COMPRESSION = "jpeg_compression"
    GAUSSIAN_BLUR = "gaussian_blur"
    RESIZE = "resize"
    GAUSSIAN_NOISE = "gaussian_noise"
    COLOR_JITTER = "color_jitter"
    CENTER_CROP = "center_crop"


# Preset parameter values exposed in the UI. Single source of truth so the
# sidebar controls and the "Randomize" picker never drift apart.
JPEG_QUALITIES: tuple[int, ...] = (90, 70, 50, 30)
BLUR_SIGMAS: tuple[float, ...] = (0.5, 1.0, 2.0)
RESIZE_SCALES: tuple[float, ...] = (0.5, 0.25)
NOISE_SIGMAS: tuple[float, ...] = (0.02, 0.05, 0.10)
COLOR_JITTER_RANGE: tuple[float, float] = (0.8, 1.2)
CENTER_CROP_FRACTION: float = 0.8


@dataclass(frozen=True)
class JpegCompressionParams:
    quality: int  # one of JPEG_QUALITIES


@dataclass(frozen=True)
class GaussianBlurParams:
    sigma: float  # one of BLUR_SIGMAS


@dataclass(frozen=True)
class ResizeParams:
    scale: float  # one of RESIZE_SCALES; downscale then upscale back to original size


@dataclass(frozen=True)
class GaussianNoiseParams:
    sigma: float  # one of NOISE_SIGMAS, applied to [0, 1]-normalized pixel values


@dataclass(frozen=True)
class ColorJitterParams:
    factor: float  # single factor within COLOR_JITTER_RANGE, applied to brightness/contrast/saturation


@dataclass(frozen=True)
class CenterCropParams:
    crop_fraction: float = CENTER_CROP_FRACTION  # crop to this fraction, then resize back to original dims


TransformParams: TypeAlias = (
    JpegCompressionParams
    | GaussianBlurParams
    | ResizeParams
    | GaussianNoiseParams
    | ColorJitterParams
    | CenterCropParams
)


@dataclass(frozen=True)
class TransformSpec:
    """A transform kind paired with its concrete parameters for a single step."""

    transform_type: TransformType
    params: TransformParams


# An ordered chain of transform steps applied to an image, e.g. Gaussian Blur
# followed by Color Jitter. Order matters — steps are applied left to right.
TransformPipeline: TypeAlias = list[TransformSpec]


TRANSFORM_DISPLAY_NAMES: dict[TransformType, str] = {
    TransformType.JPEG_COMPRESSION: "JPEG Compression",
    TransformType.GAUSSIAN_BLUR: "Gaussian Blur",
    TransformType.RESIZE: "Resize",
    TransformType.GAUSSIAN_NOISE: "Gaussian Noise",
    TransformType.COLOR_JITTER: "Color Jitter",
    TransformType.CENTER_CROP: "Center Crop",
}

TRANSFORM_DESCRIPTIONS: dict[TransformType, str] = {
    TransformType.JPEG_COMPRESSION: "Social-media re-encode, messaging",
    TransformType.GAUSSIAN_BLUR: "Out-of-focus",
    TransformType.RESIZE: "Thumbnail generation",
    TransformType.GAUSSIAN_NOISE: "Low-light sensor noise",
    TransformType.COLOR_JITTER: "Filter apps, auto-enhance",
    TransformType.CENTER_CROP: "Profile-picture cropping, framing",
}
