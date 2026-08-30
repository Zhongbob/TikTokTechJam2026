"""Inference wrapper for a trained augmentation-reversal autoencoder.

This class satisfies the `shared_types.interfaces.AutoencoderRestorer`
contract used by the app pipeline: it accepts an augmented image and returns
an image closer to the original clean version.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

from shared_types import ImagePairSample
from shared_types.interfaces import AutoencoderRestorer

from autoencoder.trainer import SimpleImageAutoencoder

SCRIPT_DIR = Path(__file__).resolve().parent


class AutoencoderRestorerImpl(AutoencoderRestorer):
    """Load a trained restoration checkpoint and restore augmented images."""

    name = "autoencoder-augmentation-reversal"
    is_placeholder = False

    DEFAULT_CHECKPOINT = SCRIPT_DIR.parent / "weights" / "autoencoder_best.pt"

    def __init__(self, model: torch.nn.Module | None = None, image_size: int = 224, device: str = "cpu") -> None:
        self.image_size = image_size
        self.device = torch.device(device)
        self._model = model
        if self._model is not None:
            self._model.to(self.device)

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, image_size: int = 224, device: str = "cpu") -> "AutoencoderRestorerImpl":
        checkpoint = torch.load(path, map_location="cpu")
        model = SimpleImageAutoencoder(hidden_channels=int(checkpoint.get("hidden_channels", 32)))
        model.load_state_dict(checkpoint["model_state"])
        return cls(model=model, image_size=int(checkpoint.get("image_size", image_size)), device=device)

    @classmethod
    def use_default(cls) -> "AutoencoderRestorerImpl":
        checkpoint = cls.DEFAULT_CHECKPOINT
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Default checkpoint not found at {checkpoint}. Train one with AutoencoderTrainer and save it there or call from_checkpoint(path)."
            )
        return cls.from_checkpoint(checkpoint)

    def restore(self, image: Image.Image, **kwargs: Any) -> Image.Image:
        if self._model is None:
            raise RuntimeError("Call load()/from_checkpoint() or initialize with a model before restore()")

        rgb = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        self._model.eval()
        with torch.no_grad():
            restored = self._model(tensor)
        restored = restored.clamp(0.0, 1.0).squeeze(0).cpu().permute(1, 2, 0).numpy()
        restored_uint8 = np.clip(restored * 255.0, 0, 255).round().astype(np.uint8)

        output = Image.fromarray(restored_uint8, mode="RGB")
        if image.size != (self.image_size, self.image_size):
            output = output.resize(image.size, Image.Resampling.BICUBIC)
        return output

    def evaluate(self, samples: Iterable[ImagePairSample], **kwargs: Any) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("Call load()/from_checkpoint() or initialize with a model before evaluate()")

        batch_size = int(kwargs.pop("batch_size", 16))
        criterion = torch.nn.MSELoss()
        from torch.utils.data import DataLoader

        from autoencoder.trainer import PairedImageDataset

        loader = DataLoader(PairedImageDataset(samples, image_size=self.image_size), batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        total_count = 0

        self._model.eval()
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device, dtype=torch.float32)
                targets = targets.to(self.device, dtype=torch.float32)
                outputs = self._model(inputs)
                total_loss += criterion(outputs, targets).item() * inputs.size(0)
                total_count += inputs.size(0)
        mse = total_loss / max(total_count, 1)
        return {"mse": float(mse), "rmse": float(np.sqrt(mse))}


# Backwards-compatible alias used by the app/service layer.
AutoencoderRestorer = AutoencoderRestorerImpl

    