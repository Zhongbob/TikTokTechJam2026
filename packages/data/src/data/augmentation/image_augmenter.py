"""Image transformations and parallel batch execution."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from image_io import load_rgb
from shared_types import AugmentationRecord, ImageInput


class ImageAugmenter:
    """Apply a random subset of six realistic transformations."""

    JPEG_QUALITIES = (90, 70, 50, 30)
    BLUR_SIGMAS = (0.5, 1.0, 2.0)
    RESIZE_SCALES = (0.5, 0.25)
    NOISE_SIGMAS = (0.02, 0.05, 0.10)
    COLOR_JITTER_RANGE = (0.8, 1.2)
    TRANSFORMS = (
        "jpeg_compression", "gaussian_blur", "resize",
        "gaussian_noise", "color_jitter", "center_crop",
    )
    CENTER_CROP_RANGE = (0.8, 1)


    def __init__(self, output_size: tuple[int, int] | None = None, seed: int | None = 42) -> None:
        self.output_size = output_size
        self.rng = np.random.default_rng(seed)

    def load_rgb(self, image: ImageInput) -> tuple[Image.Image, str]:
        return load_rgb(image, self.output_size)

    def jpeg_compression(self, image: Image.Image, quality: int) -> Image.Image:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()

    def gaussian_blur(self, image: Image.Image, sigma: float) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))

    def resize_down_up(self, image: Image.Image, scale: float) -> Image.Image:
        width, height = image.size
        small = (max(1, round(width * scale)), max(1, round(height * scale)))
        reduced = image.resize(small, Image.Resampling.LANCZOS)
        return reduced.resize((width, height), Image.Resampling.LANCZOS)

    def gaussian_noise(self, image: Image.Image, sigma: float) -> Image.Image:
        array = np.asarray(image, dtype=np.float32) / 255.0
        noise = self.rng.normal(0.0, sigma, size=array.shape).astype(np.float32)
        noisy = np.clip(array + noise, 0.0, 1.0)
        return Image.fromarray((noisy * 255.0).round().astype(np.uint8), mode="RGB")

    def color_jitter(self, image: Image.Image, brightness: float, contrast: float, saturation: float) -> Image.Image:
        result = ImageEnhance.Brightness(image).enhance(brightness)
        result = ImageEnhance.Contrast(result).enhance(contrast)
        return ImageEnhance.Color(result).enhance(saturation)

    def center_crop(self, image: Image.Image, crop_ratio: float = 0.8) -> Image.Image:
        width, height = image.size
        crop_width, crop_height = max(1, round(width * crop_ratio)), max(1, round(height * crop_ratio))
        left, top = (width - crop_width) // 2, (height - crop_height) // 2
        cropped = image.crop((left, top, left + crop_width, top + crop_height))
        return cropped.resize((width, height), Image.Resampling.LANCZOS), {"crop_ratio": crop_ratio}

    def _apply_random_parameters(self, image: Image.Image, transform: str) -> tuple[Image.Image, dict[str, Any]]:
        if transform == "jpeg_compression":
            quality = int(self.rng.choice(self.JPEG_QUALITIES))
            return self.jpeg_compression(image, quality), {"quality": quality}
        if transform == "gaussian_blur":
            sigma = float(self.rng.choice(self.BLUR_SIGMAS))
            return self.gaussian_blur(image, sigma), {"sigma": sigma}
        if transform == "resize":
            scale = float(self.rng.choice(self.RESIZE_SCALES))
            return self.resize_down_up(image, scale), {"scale": scale}
        if transform == "gaussian_noise":
            sigma = float(self.rng.choice(self.NOISE_SIGMAS))
            return self.gaussian_noise(image, sigma), {"sigma": sigma}
        if transform == "color_jitter":
            low, high = self.COLOR_JITTER_RANGE
            brightness, contrast, saturation = map(float, self.rng.uniform(low, high, size=3))
            output = self.color_jitter(image, brightness, contrast, saturation)
            return output, {"brightness": brightness, "contrast": contrast, "saturation": saturation}
        if transform == "center_crop":
            return self.center_crop(image, crop_ratio = float(self.rng.uniform(*self.CENTER_CROP_RANGE)))
        raise ValueError(f"Unknown transform: {transform}")

    def transform_one(self, image: ImageInput, num_augmentations: int = 6) -> tuple[np.ndarray, AugmentationRecord]:
        if not 1 <= num_augmentations <= len(self.TRANSFORMS):
            raise ValueError("num_augmentations must be between 1 and 6")
        transformed, source = self.load_rgb(image)
        order = list(self.TRANSFORMS)
        self.rng.shuffle(order)
        selected = order[:num_augmentations]
        steps = []
        for name in selected:
            transformed, parameters = self._apply_random_parameters(transformed, name)
            steps.append({"transform": name, "parameters": parameters})
        record = AugmentationRecord(source, f"random_{num_augmentations}_of_6", {
            "num_augmentations": num_augmentations,
            "available_transforms": list(self.TRANSFORMS),
            "order": selected,
            "steps": steps,
        })
        return np.asarray(transformed, dtype=np.uint8), record

    def transform_images(
        self,
        images: Sequence[ImageInput],
        num_augmentations: int = 6,
        return_metadata: bool = False,
        backend: str = "sequential",
        num_workers: int | None = None,
    ):
        """Transform images in input order using sequential, thread, or process workers."""
        if backend not in {"sequential", "thread", "process"}:
            raise ValueError("backend must be sequential, thread, or process")
        if num_workers is not None and num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        images = list(images)
        if not images:
            output = np.empty((0,), dtype=np.uint8)
            return (output, []) if return_metadata else output

        if backend == "sequential" or num_workers == 1:
            results = [self.transform_one(image, num_augmentations) for image in images]
        else:
            seeds = self.rng.integers(0, np.iinfo(np.uint32).max, size=len(images), dtype=np.uint32)
            payloads = [(image, self.output_size, int(seed), num_augmentations) for image, seed in zip(images, seeds)]
            executor_type = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
            with executor_type(max_workers=num_workers) as executor:
                results = list(executor.map(_transform_worker, payloads))

        arrays = [array for array, _ in results]
        records = [record for _, record in results]
        output = np.stack(arrays) if len({array.shape for array in arrays}) == 1 else arrays
        return (output, records) if return_metadata else output


def _transform_worker(payload: tuple[ImageInput, tuple[int, int] | None, int, int]) -> tuple[np.ndarray, AugmentationRecord]:
    """Top-level worker required by Python multiprocessing."""
    image, output_size, seed, num_augmentations = payload
    return ImageAugmenter(output_size, seed).transform_one(image, num_augmentations)
