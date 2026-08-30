"""Reusable image loading and discovery helpers."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
from shared_types import ImageInput

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def load_rgb(image: ImageInput, output_size: tuple[int, int] | None = None) -> tuple[Image.Image, str]:
    """Load a supported input as RGB and optionally resize it."""
    if isinstance(image, (str, Path)):
        path = Path(image)
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Expected PNG/JPEG, received: {path}")
        with Image.open(path) as opened:
            pil_image = opened.convert("RGB")
        source = str(path)
    elif isinstance(image, (bytes, bytearray, memoryview)):
        with Image.open(BytesIO(bytes(image))) as opened:
            pil_image = opened.convert("RGB")
        source = "<bytes>"
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
        source = "<PIL.Image>"
    elif isinstance(image, np.ndarray):
        array = image
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and array.max() <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(array).convert("RGB")
        source = "<numpy.ndarray>"
    else:
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    if output_size is not None:
        pil_image = pil_image.resize(output_size, Image.Resampling.LANCZOS)
    return pil_image, source


def find_images(directory: str | Path, recursive: bool = True) -> list[Path]:
    """Return sorted PNG/JPEG paths from a directory."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {directory}")
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    paths = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not paths:
        raise ValueError(f"No PNG/JPEG images found in: {directory}")
    return paths
