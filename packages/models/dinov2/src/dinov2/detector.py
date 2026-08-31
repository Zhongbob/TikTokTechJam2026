"""Inference wrapper for the in-house fine-tuned DINOv2 AIGC classifier.

The checkpoint (``dino*.pt`` at the repo root) is::

    torch.save({
        "model_name": "facebook/dinov2-small",     # or -base / -large
        "num_classes": 2,
        "class_names": {0: "real", 1: "synthetic"},
        "model_state_dict": {                       # backbone.* + classifier.*
            "backbone.<HF Dinov2Model keys>": ...,
            "classifier.1.weight": [num_classes, hidden],
            "classifier.1.bias":   [num_classes],
        },
        "val_metrics": {...}, "config": {...},
    }, path)

Architecture: a HF ``Dinov2Model`` backbone + ``Sequential(Dropout, Linear)`` on
the CLS token (``last_hidden_state[:, 0]``). Preprocessing uses that model's HF
image processor (resize shortest edge to 256, centre-crop 224, ImageNet norm).

`torch` / `transformers` are imported lazily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from detector_common import ImageDetector, locate_checkpoint, resolve_device
from detector_common.weights import candidate_weight_dirs
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
#: bundled checkpoint dir, then repo root (dino.pt is ~253 MiB so it usually
#: lives in the repo root, not here)
DEFAULT_WEIGHTS_DIR = SCRIPT_DIR.parent / "weights"
#: packages/models/dinov2/src/dinov2 -> repo root
REPO_ROOT = SCRIPT_DIR.parents[4]

#: Project HF bucket the default ``dino.pt`` is pulled from when it isn't already
#: on disk (it's ~253 MiB, so it's git-ignored rather than committed).
HF_BUCKET_ID = "Zhongbob2/TikTokTechJam"
HF_BUCKET_CHECKPOINT = "dino.pt"

DEFAULT_MODEL_NAME = "facebook/dinov2-small"
_DEFAULT_CLASS_NAMES = {0: "real", 1: "synthetic"}
_AI_HINTS = ("synthetic", "fake", "ai", "generated", "gan", "diffusion", "tampered", "aigc", "deepfake")
_REAL_HINTS = ("real", "authentic", "natural", "genuine", "pristine")
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _int_keyed(names: Any) -> dict[int, str]:
    if not isinstance(names, dict):
        return {i: str(n) for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def _resolve_positive_index(class_names: dict[int, str], override: int | str | None) -> int:
    if isinstance(override, int):
        return override
    if isinstance(override, str):
        for index, name in class_names.items():
            if override.lower() in name.lower():
                return index
        raise ValueError(f"positive_class={override!r} not in {list(class_names.values())}")
    real = [i for i, n in class_names.items() if any(h in n.lower() for h in _REAL_HINTS)]
    if len(real) == 1 and len(class_names) == 2:
        return next(i for i in class_names if i != real[0])
    ai = [i for i, n in class_names.items() if any(h in n.lower() for h in _AI_HINTS)]
    if len(ai) == 1:
        return ai[0]
    return max(class_names)


def _download_default_checkpoint(dest_dir: Path) -> Path:
    """Fetch the default ``dino.pt`` from the project's HF bucket into ``dest_dir``.

    Used by `DINOv2Detector.use_default` when no checkpoint is found locally.
    """
    from huggingface_hub import download_bucket_files

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / HF_BUCKET_CHECKPOINT
    print(
        f"[dinov2] no local checkpoint — downloading {HF_BUCKET_CHECKPOINT} from "
        f"hf.co/buckets/{HF_BUCKET_ID} -> {dest}"
    )
    download_bucket_files(
        bucket_id=HF_BUCKET_ID,
        files=[(HF_BUCKET_CHECKPOINT, dest)],
        raise_on_missing_files=True,
    )
    return dest


def _build_dinov2_classifier(model_name: str, num_classes: int, dropout: float) -> Any:
    """HF ``Dinov2Model`` backbone + ``Sequential(Dropout, Linear)`` CLS head."""
    from torch import nn
    from transformers import Dinov2Model

    class DinoV2Classifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = Dinov2Model.from_pretrained(model_name)
            hidden = self.backbone.config.hidden_size
            self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_classes))

        def forward(self, pixel_values: Any) -> Any:
            outputs = self.backbone(pixel_values=pixel_values)
            cls_token = outputs.last_hidden_state[:, 0]
            return self.classifier(cls_token)

    return DinoV2Classifier()


class DINOv2Detector(ImageDetector):
    """Fine-tuned DINOv2 + linear head, wrapped as an `ImageDetector`.

    ``positive_class`` overrides which class means "AI-generated" (int index or a
    class-name substring); by default inferred from the checkpoint's
    ``class_names``. ``flip=True`` inverts the final score.
    """

    name = "dinov2-aigc"
    is_placeholder = False

    def __init__(
        self,
        model: Any,
        *,
        class_names: dict[int, str],
        processor: Any | None = None,
        image_size: int = 224,
        device: str = "auto",
        positive_class: int | str | None = None,
        flip: bool = False,
    ) -> None:
        import torch

        self.device = resolve_device(device)
        self._torch = torch
        self._model = model.to(self.device).eval()
        self._processor = processor
        self.class_names = class_names
        self._positive_index = _resolve_positive_index(class_names, positive_class)
        self.flip = flip

        if processor is None:
            from torchvision import transforms

            resize = int(round(image_size / 0.875))
            self._transform = transforms.Compose([
                transforms.Resize(resize, antialias=True),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ])
        else:
            self._transform = None

    # --- construction --------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "auto",
        positive_class: int | str | None = None,
        dropout: float = 0.0,
        flip: bool = False,
    ) -> "DINOv2Detector":
        import torch

        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        meta = blob if isinstance(blob, dict) else {}
        state = meta.get("model_state_dict") or meta.get("state_dict") or blob

        model_name = meta.get("model_name", DEFAULT_MODEL_NAME)
        class_names = _int_keyed(meta.get("class_names") or _DEFAULT_CLASS_NAMES)
        num_classes = int(meta.get("num_classes", len(class_names)))

        model = _build_dinov2_classifier(model_name, num_classes, dropout)
        state = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state.items()
            if hasattr(v, "shape")
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        target = len(model.state_dict())
        print(f"[dinov2] loaded {Path(path).name} ({model_name}): "
              f"{len(missing)}/{target} missing, {len(unexpected)} unexpected")
        head_missing = [k for k in missing if k.startswith("classifier")]
        if head_missing or len(missing) > 0.1 * target:
            print(f"[dinov2] WARNING: {len(missing)} unfilled weights ({head_missing} in the head) "
                  "-- check model_name / num_classes.")

        processor = None
        try:
            from transformers import AutoImageProcessor

            processor = AutoImageProcessor.from_pretrained(model_name)
        except Exception:  # noqa: BLE001 - fall back to the torchvision transform
            pass

        return cls(
            model, class_names=class_names, processor=processor,
            image_size=int(meta.get("image_size", 224)),
            device=device, positive_class=positive_class, flip=flip,
        )

    from_pretrained = from_checkpoint

    @classmethod
    def use_default(
        cls,
        *,
        device: str = "auto",
        checkpoint: str | Path | None = None,
        **kwargs: Any,
    ) -> "DINOv2Detector":
        """``checkpoint=`` takes an explicit ``.pt`` path; otherwise ``dino*.pt``
        is searched for across the usual weight locations
        (``$DINOV2_CHECKPOINT``, the package ``weights/``, the repo checkout, the
        cwd, ``/content``). If nothing is found, the default checkpoint is
        downloaded from the project's HF bucket into the package ``weights/``
        folder."""
        if checkpoint is not None:
            return cls.from_checkpoint(Path(checkpoint).expanduser(), device=device, **kwargs)
        hit = locate_checkpoint(
            ("dino.pt", "dino*.pt", "dinov2*.pt", "dino*.pth"),
            script_dir=SCRIPT_DIR, env_var="DINOV2_CHECKPOINT",
        )
        if hit is None:
            try:
                hit = _download_default_checkpoint(DEFAULT_WEIGHTS_DIR)
            except Exception as error:  # noqa: BLE001 - offline / HF unavailable
                looked = ", ".join(
                    str(d) for d in candidate_weight_dirs(SCRIPT_DIR, env_var="DINOV2_CHECKPOINT")
                )
                raise FileNotFoundError(
                    "DINOv2 checkpoint (dino.pt) not found and the automatic "
                    f"download failed ({error}). Pass checkpoint=<path>, set "
                    f"$DINOV2_CHECKPOINT, or drop it in one of: {looked}"
                ) from error
        return cls.from_checkpoint(hit, device=device, **kwargs)

    # --- scoring -----------------------------------------------------

    def _pixel_values(self, image: Image.Image) -> Any:
        rgb = image.convert("RGB")
        if self._processor is not None:
            return self._processor(images=rgb, return_tensors="pt")["pixel_values"].to(self.device)
        return self._transform(rgb).unsqueeze(0).to(self.device)

    def raw_output(self, image: Image.Image) -> Any:
        torch = self._torch
        with torch.no_grad():
            return self._model(self._pixel_values(image)).float().reshape(-1)

    def _score(self, image: Image.Image) -> float:
        torch = self._torch
        logits = self.raw_output(image)
        if logits.numel() == 1:
            p_ai = 1.0 / (1.0 + float(torch.exp(-logits[0]).item()))
        else:
            p_ai = float(torch.softmax(logits, dim=-1)[self._positive_index].item())
        return 1.0 - p_ai if self.flip else p_ai
