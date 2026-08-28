"""Image transformations and randomized all-five augmentation permutations."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from libs.shared_types.augmentation import AugmentationRecord, ImageInput


class ImageAugmenter:
    """Apply all five hackathon transformations once in a randomized order."""

    JPEG_QUALITIES = (90, 70, 50, 30)
    BLUR_SIGMAS = (0.5, 1.0, 2.0)
    RESIZE_SCALES = (0.5, 0.25)
    NOISE_SIGMAS = (0.02, 0.05, 0.10)
    COLOR_JITTER_RANGE = (0.8, 1.2)
    TRANSFORMS = (
        "jpeg_compression",
        "gaussian_blur",
        "resize",
        "gaussian_noise",
        "color_jitter",
    )

    def __init__(
        self,
        output_size: tuple[int, int] | None = None,
        seed: int | None = 42,
    ) -> None:
        """Create an augmenter.

        Args:
            output_size: Optional `(width, height)` applied before augmentation.
            seed: Random seed. Use `None` for non-reproducible results.
        """
        self.output_size = output_size
        self.rng = np.random.default_rng(seed)

    def load_rgb(self, image: ImageInput) -> tuple[Image.Image, str]:
        """Load a path, PIL image, or NumPy array as an RGB PIL image."""
        if isinstance(image, (str, Path)):
            path = Path(image)
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
                raise ValueError(f"Expected PNG/JPEG, received: {path}")
            with Image.open(path) as opened:
                pil_image = opened.convert("RGB")
            source = str(path)
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

        if self.output_size is not None:
            pil_image = pil_image.resize(self.output_size, Image.Resampling.LANCZOS)
        return pil_image, source

    def jpeg_compression(self, image: Image.Image, quality: int) -> Image.Image:
        """JPEG encode and decode an image at an allowed quality."""
        if quality not in self.JPEG_QUALITIES:
            raise ValueError(f"quality must be one of {self.JPEG_QUALITIES}")
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()

    def gaussian_blur(self, image: Image.Image, sigma: float) -> Image.Image:
        """Apply Gaussian blur using an allowed sigma."""
        if sigma not in self.BLUR_SIGMAS:
            raise ValueError(f"sigma must be one of {self.BLUR_SIGMAS}")
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))

    def resize_down_up(self, image: Image.Image, scale: float) -> Image.Image:
        """Downscale by an allowed factor, then restore the original dimensions."""
        if scale not in self.RESIZE_SCALES:
            raise ValueError(f"scale must be one of {self.RESIZE_SCALES}")
        width, height = image.size
        reduced_size = (
            max(1, round(width * scale)),
            max(1, round(height * scale)),
        )
        reduced = image.resize(reduced_size, Image.Resampling.LANCZOS)
        return reduced.resize((width, height), Image.Resampling.LANCZOS)

    def gaussian_noise(self, image: Image.Image, sigma: float) -> Image.Image:
        """Add zero-mean Gaussian noise in normalized pixel space."""
        if sigma not in self.NOISE_SIGMAS:
            raise ValueError(f"sigma must be one of {self.NOISE_SIGMAS}")
        array = np.asarray(image, dtype=np.float32) / 255.0
        noise = self.rng.normal(0.0, sigma, size=array.shape).astype(np.float32)
        noisy = np.clip(array + noise, 0.0, 1.0)
        return Image.fromarray((noisy * 255.0).round().astype(np.uint8))

    def color_jitter(
        self,
        image: Image.Image,
        brightness: float,
        contrast: float,
        saturation: float,
    ) -> Image.Image:
        """Apply brightness, contrast, and saturation changes within ±20%."""
        low, high = self.COLOR_JITTER_RANGE
        values = (brightness, contrast, saturation)
        if not all(low <= value <= high for value in values):
            raise ValueError("Color-jitter factors must be within [0.8, 1.2]")
        result = ImageEnhance.Brightness(image).enhance(brightness)
        result = ImageEnhance.Contrast(result).enhance(contrast)
        return ImageEnhance.Color(result).enhance(saturation)

    def _apply_random_parameters(
        self,
        image: Image.Image,
        transform: str,
    ) -> tuple[Image.Image, dict[str, Any]]:
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
            brightness, contrast, saturation = self.rng.uniform(low, high, size=3)
            parameters = {
                "brightness": float(brightness),
                "contrast": float(contrast),
                "saturation": float(saturation),
            }
            result = self.color_jitter(image, **parameters)
            return result, parameters
        raise ValueError(f"Unknown transform {transform!r}")

    def transform_one(
        self,
        image: ImageInput,
    ) -> tuple[np.ndarray, AugmentationRecord]:
        """Apply all transforms exactly once, shuffling their order."""
        clean, source = self.load_rgb(image)
        order = list(self.TRANSFORMS)
        self.rng.shuffle(order)

        transformed = clean
        steps: list[dict[str, Any]] = []
        for transform in order:
            transformed, parameters = self._apply_random_parameters(
                transformed,
                transform,
            )
            steps.append({"transform": transform, "parameters": parameters})

        record = AugmentationRecord(
            source=source,
            transform="permutation_all_5",
            parameters={"order": order, "steps": steps},
        )
        return np.asarray(transformed, dtype=np.uint8), record

    @staticmethod
    def _stack_or_list(
        images: list[np.ndarray],
    ) -> np.ndarray | list[np.ndarray]:
        if not images:
            return np.empty((0,), dtype=np.uint8)
        shapes = {image.shape for image in images}
        return np.stack(images) if len(shapes) == 1 else images

    def transform_images(
        self,
        images: Sequence[ImageInput],
        return_metadata: bool = False,
    ) -> (
        np.ndarray
        | list[np.ndarray]
        | tuple[np.ndarray | list[np.ndarray], list[AugmentationRecord]]
    ):
        """Transform each input image with an independent all-five permutation."""
        transformed: list[np.ndarray] = []
        records: list[AugmentationRecord] = []
        for image in images:
            result, record = self.transform_one(image)
            transformed.append(result)
            records.append(record)
        output = self._stack_or_list(transformed)
        return (output, records) if return_metadata else output

    def transform_and_save(
        self,
        image_paths: Sequence[str | Path],
        output_dir: str | Path,
        suffix: str = "_augmented",
    ) -> tuple[
        np.ndarray | list[np.ndarray],
        list[str],
        list[AugmentationRecord],
    ]:
        """Transform files and save new PNGs without changing the originals."""
        transformed, records = self.transform_images(
            image_paths,
            return_metadata=True,
        )
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        saved_paths: list[str] = []

        for source, array in zip(image_paths, list(transformed)):
            source_path = Path(source)
            destination = output_path / f"{source_path.stem}{suffix}.png"
            counter = 1
            while destination.exists():
                destination = output_path / (
                    f"{source_path.stem}{suffix}_{counter}.png"
                )
                counter += 1
            Image.fromarray(array).save(destination, format="PNG")
            saved_paths.append(str(destination))

        return transformed, saved_paths, records
