"""Inference wrapper for the in-house fine-tuned ConvNeXt AIGC classifier.

The checkpoint is a standard Hugging Face `ConvNextForImageClassification`
directory (``config.json`` + ``model.safetensors`` + ``preprocessor_config.json``)
trained on real-vs-AI-generated images, ``id2label = {0: "real", 1: "synthetic"}``,
224px, ImageNet normalisation. It ships as ``convnext_aigc_run/best_model/``
inside ``convnext_aigc_run*.zip`` at the repo root; `from_checkpoint` accepts the
``.zip``, the extracted run dir, or the ``best_model`` dir directly.

    detector = ConvNextAIGCDetector.use_default()
    result   = detector.predict(pil_image)
    metrics  = detector.evaluate(val_samples, generate_confusion_matrix=True)

`torch` / `transformers` are imported lazily.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path
from typing import Any

from detector_common import ImageDetector, resolve_device
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
#: bundled checkpoint dir (populated if the .zip is < 100 MiB) then repo root
DEFAULT_WEIGHTS_DIR = SCRIPT_DIR.parent / "weights"
#: packages/models/convnext_aigc/src/convnext_aigc -> repo root
REPO_ROOT = SCRIPT_DIR.parents[4]

_AI_HINTS = ("synthetic", "fake", "ai", "generated", "gan", "diffusion", "tampered", "aigc")


def _resolve_model_dir(path: str | Path) -> Path:
    """Return a HF model dir (has ``config.json`` + weights) given the ``.zip``,
    the run dir, or the ``best_model`` dir."""
    path = Path(path)
    if path.is_dir():
        if (path / "config.json").is_file():
            return path
        for candidate in (path / "best_model", *path.glob("*/best_model")):
            if (candidate / "config.json").is_file():
                return candidate
        raise FileNotFoundError(f"no HF model dir (config.json) under {path}")
    if path.suffix.lower() == ".zip":
        dest = Path(tempfile.mkdtemp(prefix="convnext_aigc_"))
        with zipfile.ZipFile(path) as archive:
            archive.extractall(dest)
        for config in dest.rglob("config.json"):
            if (config.parent / "model.safetensors").is_file() or (config.parent / "pytorch_model.bin").is_file():
                return config.parent
        raise FileNotFoundError(f"no HF model dir found inside {path}")
    raise FileNotFoundError(f"{path} is neither a model directory nor a .zip")


class ConvNextAIGCDetector(ImageDetector):
    """Fine-tuned ConvNeXt image classifier, wrapped as an `ImageDetector`."""

    name = "convnext-aigc"
    is_placeholder = False

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        device: str = "auto",
        positive_index: int | None = None,
    ) -> None:
        import torch

        self.device = resolve_device(device)
        self._torch = torch
        self._model = model.to(self.device).eval()
        self._processor = processor
        self._positive_index = self._resolve_positive_index(model, positive_index)

    # --- construction --------------------------------------------------

    @staticmethod
    def _resolve_positive_index(model: Any, override: int | None) -> int:
        if override is not None:
            return int(override)
        id2label = getattr(getattr(model, "config", None), "id2label", None) or {}
        for index, label in id2label.items():
            if any(hint in str(label).lower() for hint in _AI_HINTS):
                return int(index)
        return 1 if len(id2label) >= 2 else 0

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "auto",
        positive_index: int | None = None,
    ) -> "ConvNextAIGCDetector":
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        model_dir = _resolve_model_dir(path)
        model = AutoModelForImageClassification.from_pretrained(str(model_dir))
        processor = AutoImageProcessor.from_pretrained(str(model_dir))
        return cls(model, processor, device=device, positive_index=positive_index)

    from_pretrained = from_checkpoint

    @classmethod
    def use_default(
        cls,
        *,
        device: str = "auto",
        checkpoint: str | Path | None = None,
        positive_index: int | None = None,
    ) -> "ConvNextAIGCDetector":
        """``checkpoint=`` takes an explicit ``.zip`` / model dir; otherwise the
        first ``convnext_aigc*`` in ``src/weights/`` then the repo root."""
        if checkpoint is not None:
            return cls.from_checkpoint(Path(checkpoint).expanduser(), device=device,
                                       positive_index=positive_index)
        for directory in (DEFAULT_WEIGHTS_DIR, REPO_ROOT):
            for pattern in ("convnext_aigc*", "convnext*aigc*"):
                for hit in sorted(directory.glob(pattern)):
                    if hit.is_dir() or hit.suffix.lower() == ".zip":
                        return cls.from_checkpoint(hit, device=device, positive_index=positive_index)
        raise FileNotFoundError(
            f"No ConvNeXt AIGC checkpoint (a run dir or *.zip) found in "
            f"{DEFAULT_WEIGHTS_DIR} or {REPO_ROOT}. Pass checkpoint=<path>, put "
            "convnext_aigc_run.zip in one of them, or call from_checkpoint(path)."
        )

    # --- scoring -----------------------------------------------------

    def raw_output(self, image: Image.Image) -> Any:
        torch = self._torch
        inputs = self._processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            return self._model(**inputs).logits.float().squeeze(0)

    def _score(self, image: Image.Image) -> float:
        torch = self._torch
        logits = self.raw_output(image)
        if logits.ndim == 0 or logits.shape[-1] == 1:
            return float(torch.sigmoid(logits.reshape(())).item())
        return float(torch.softmax(logits, dim=-1)[self._positive_index].item())
