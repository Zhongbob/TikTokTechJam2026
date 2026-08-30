"""Inference wrapper around the Community-Forensics ViT detector.

Source: JeongsooP/Community-Forensics (official; training + eval pipeline and
checkpoint links). HF mirror used here by default:
``buildborderless/CommunityForensics-DeepfakeDet-ViT`` (ViT-Small, loads with
``transformers``).

`heavy` deps (`torch`, `transformers`) are imported lazily inside the methods
that need them, so importing this module for tooling stays cheap.

NOTE ON LABEL ORDER: the wrapper reads ``model.config.id2label`` to find the
"AI-generated" class. If the loaded checkpoint has unlabelled classes it falls
back to `positive_index` (default 1). Set that / `positive_index=` explicitly
if scores look inverted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from detector_common import ImageDetector, resolve_device
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent

#: Default source — a Hugging Face repo id (downloaded + cached on first use)
#: or a local directory saved with ``model.save_pretrained(...)``.
DEFAULT_MODEL = "buildborderless/CommunityForensics-DeepfakeDet-ViT"

#: Where `use_default()` looks for an offline copy before hitting the Hub.
DEFAULT_LOCAL_DIR = SCRIPT_DIR.parent / "weights" / "community_forensics"

_AI_LABEL_HINTS = ("fake", "synthetic", "ai", "generated", "gan", "diffusion")


class CommunityForensicsDetector(ImageDetector):
    """Community-Forensics ViT, wrapped as an `ImageDetector`.

        detector = CommunityForensicsDetector.use_default()
        result = detector.predict(pil_image)
        metrics = detector.evaluate(val_samples, generate_confusion_matrix=True)
    """

    name = "community-forensics-vit"
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
        self._model = model.to(self.device).eval()
        self._processor = processor
        self._torch = torch
        self._positive_index = self._resolve_positive_index(model, positive_index)

    # --- construction --------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path = DEFAULT_MODEL,
        *,
        device: str = "auto",
        positive_index: int | None = None,
        **hf_kwargs: Any,
    ) -> "CommunityForensicsDetector":
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        name = str(model_name_or_path)
        model = AutoModelForImageClassification.from_pretrained(name, **hf_kwargs)
        try:
            processor = AutoImageProcessor.from_pretrained(name)
        except Exception:  # some repos ship only the model; use a plain ViT processor
            processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        return cls(model, processor, device=device, positive_index=positive_index)

    # keep the naming consistent with the other detectors in this repo
    from_checkpoint = from_pretrained

    @classmethod
    def use_default(cls, *, device: str = "auto") -> "CommunityForensicsDetector":
        """Load the default model — an offline copy under
        ``src/weights/community_forensics/`` if present, else download
        ``DEFAULT_MODEL`` from the Hugging Face Hub."""
        source = DEFAULT_LOCAL_DIR if DEFAULT_LOCAL_DIR.is_dir() else DEFAULT_MODEL
        return cls.from_pretrained(source, device=device)

    # --- scoring -----------------------------------------------------

    @staticmethod
    def _resolve_positive_index(model: Any, override: int | None) -> int:
        if override is not None:
            return override
        id2label = getattr(getattr(model, "config", None), "id2label", None) or {}
        for index, label in id2label.items():
            if any(hint in str(label).lower() for hint in _AI_LABEL_HINTS):
                return int(index)
        return 1 if len(id2label) >= 2 else 0

    def _score(self, image: Image.Image) -> float:
        torch = self._torch
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits.float().squeeze(0)  # (C,) or scalar

        if logits.ndim == 0 or logits.shape[-1] == 1:
            # single-logit head: value is the AI-generated logit
            return float(torch.sigmoid(logits.reshape(())).item())
        probs = torch.softmax(logits, dim=-1)
        return float(probs[self._positive_index].item())
