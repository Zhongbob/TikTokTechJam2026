"""Image transformations and parallel batch execution."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

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

    def _resolve_num_augmentations(self, num_augmentations: int | tuple[int, int]) -> int:
        """Accept a fixed count or an inclusive ``(min, max)`` range; when a
        range is given, draw a per-call count uniformly from it so different
        images get random-length transform chains.

        ``0`` is allowed -- the image is passed through unchanged (bar the
        ``output_size`` resize), so ``(0, 6)`` mixes clean and corrupted images.
        """
        if isinstance(num_augmentations, int):
            low = high = num_augmentations
        else:
            low, high = num_augmentations
        if not 0 <= low <= high <= len(self.TRANSFORMS):
            raise ValueError(
                "num_augmentations must be an int in 0..6, or a (min, max) pair within that range"
            )
        return int(self.rng.integers(low, high + 1))

    def transform_one(
        self, image: ImageInput, num_augmentations: int | tuple[int, int] = 6
    ) -> tuple[np.ndarray, AugmentationRecord]:
        count = self._resolve_num_augmentations(num_augmentations)
        transformed, source = self.load_rgb(image)
        order = list(self.TRANSFORMS)
        self.rng.shuffle(order)
        selected = order[:count]
        steps = []
        for name in selected:
            transformed, parameters = self._apply_random_parameters(transformed, name)
            steps.append({"transform": name, "parameters": parameters})
        record = AugmentationRecord(source, f"random_{count}_of_6", {
            "num_augmentations": count,
            "requested_num_augmentations": num_augmentations,
            "available_transforms": list(self.TRANSFORMS),
            "order": selected,
            "steps": steps,
        })
        return np.asarray(transformed, dtype=np.uint8), record

    def transform_images(
        self,
        images: Sequence[ImageInput],
        num_augmentations: int | tuple[int, int] = 6,
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

    def iter_transform_images(
        self,
        images: Iterable[ImageInput],
        num_augmentations: int | tuple[int, int] = 6,
        return_metadata: bool = False,
        backend: str = "sequential",
        num_workers: int | None = None,
        prefetch: int | None = None,
    ) -> Iterator:
        """Stream transformed images in input order, one result at a time.

        Behaves like :meth:`transform_images` but never materialises the whole
        dataset: it consumes ``images`` lazily and yields either a single
        ``np.ndarray`` per image, or ``(np.ndarray, AugmentationRecord)`` when
        ``return_metadata`` is true. For the ``thread``/``process`` backends at
        most ``prefetch`` items (default: ``num_workers`` or 4) are kept in
        flight, bounding peak memory regardless of dataset size.
        """
        if backend not in {"sequential", "thread", "process"}:
            raise ValueError("backend must be sequential, thread, or process")
        if num_workers is not None and num_workers < 1:
            raise ValueError("num_workers must be at least 1")

        def emit(result):
            array, record = result
            return (array, record) if return_metadata else array

        if backend == "sequential" or num_workers == 1:
            for image in images:
                yield emit(self.transform_one(image, num_augmentations))
            return

        window = prefetch if prefetch is not None else (num_workers or 4)
        if window < 1:
            raise ValueError("prefetch must be at least 1")
        executor_type = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
        pending: deque = deque()
        with executor_type(max_workers=num_workers) as executor:
            for image in images:
                seed = int(self.rng.integers(0, np.iinfo(np.uint32).max, dtype=np.uint32))
                payload = (image, self.output_size, seed, num_augmentations)
                pending.append(executor.submit(_transform_worker, payload))
                if len(pending) >= window:
                    yield emit(pending.popleft().result())
            while pending:
                yield emit(pending.popleft().result())


def _transform_worker(
    payload: tuple[ImageInput, tuple[int, int] | None, int, int | tuple[int, int]],
) -> tuple[np.ndarray, AugmentationRecord]:
    """Top-level worker required by Python multiprocessing."""
    image, output_size, seed, num_augmentations = payload
    return ImageAugmenter(output_size, seed).transform_one(image, num_augmentations)
