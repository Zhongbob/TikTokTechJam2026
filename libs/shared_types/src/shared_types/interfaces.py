from __future__ import annotations

from typing import Protocol, runtime_checkable

from PIL import Image

from shared_types.detection import DetectionResult


@runtime_checkable
class AutoencoderRestorer(Protocol):
    """Restores a transformed image back towards its original, undegraded form.

    Implemented as a structural Protocol so the real model in
    packages/models/autoencoder can satisfy this contract without importing
    anything from this library — it just needs matching attributes/methods.
    """

    name: str
    is_placeholder: bool

    def restore(self, image: Image.Image) -> Image.Image: ...


@runtime_checkable
class EnsembleDetector(Protocol):
    """Classifies an image as real or AI-generated."""

    name: str
    is_placeholder: bool

    def predict(self, image: Image.Image) -> DetectionResult: ...
