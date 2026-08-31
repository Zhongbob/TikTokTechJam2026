"""Inference wrapper around the fine-tuned Swin-Tiny AIGC classifier.

Loads the ``.pth`` written by ``s3_swin_transformer_streaming.ipynb``:

    torch.save({
        "model_name": "microsoft_swin_tiny_patch4_window7_224",
        "model_state_dict": model.state_dict(),   # microsoft/Swin-Transformer SwinTransformer
        "class_names": {0: "real", 1: "synthetic"},
        "image_size": 224,
        "imagenet_mean": (0.485, 0.456, 0.406),
        "imagenet_std": (0.229, 0.224, 0.225),
        ...
    }, path)

The architecture is rebuilt from the vendored ``_swin_transformer.py`` (same
Swin-Tiny config the notebook trains) so the state dict loads exactly. A bare
``state_dict`` (just the tensors) is also accepted — defaults are then assumed
for class names / image size / normalisation.

``torch`` / ``torchvision`` are imported lazily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from detector_common import ImageDetector, locate_checkpoint, resolve_device
from detector_common.weights import candidate_weight_dirs
from PIL import Image

from swin._swin_transformer import SwinTransformer

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR.parent / "weights" / "best_swin_tiny_binary.pth"

# Swin-Tiny (matches build_swin_tiny() in the training notebook).
_SWIN_TINY = dict(
    patch_size=4, in_chans=3, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
    window_size=7, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
    drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.2, ape=False, patch_norm=True,
)
_DEFAULT_CLASS_NAMES = {0: "real", 1: "synthetic"}
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_AI_HINTS = ("synthetic", "fake", "ai", "generated", "gan", "diffusion", "aigc", "deepfake")
_REAL_HINTS = ("real", "authentic", "natural", "genuine", "pristine")


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
    return max(class_names)  # fall back to the 0=real, 1=positive convention


class SwinDetector(ImageDetector):
    """Fine-tuned Swin-Tiny, wrapped as an `ImageDetector`.

        detector = SwinDetector.use_default()                      # bundled checkpoint
        detector = SwinDetector.from_checkpoint("best_swin_tiny_binary.pth")
        result = detector.predict(pil_image)
        metrics = detector.evaluate(val_samples, generate_confusion_matrix=True)

    ``positive_class`` overrides which class means "AI-generated" (int index or
    class-name substring); by default it's inferred from the checkpoint's
    ``class_names`` (``synthetic`` here).
    """

    name = "swin-tiny-aigc"
    is_placeholder = False

    def __init__(
        self,
        model: Any,
        *,
        class_names: dict[int, str] | None = None,
        image_size: int = 224,
        mean: tuple[float, float, float] = _IMAGENET_MEAN,
        std: tuple[float, float, float] = _IMAGENET_STD,
        device: str = "auto",
        positive_class: int | str | None = None,
    ) -> None:
        import torch
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        self.device = resolve_device(device)
        self._torch = torch
        self._model = model.to(self.device).eval()
        self.class_names = class_names or dict(_DEFAULT_CLASS_NAMES)
        self._positive_index = _resolve_positive_index(self.class_names, positive_class)
        self._transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    # --- construction --------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "auto",
        positive_class: int | str | None = None,
    ) -> "SwinDetector":
        import torch
        from torch import nn

        blob = torch.load(str(path), map_location="cpu", weights_only=False)

        meta: dict[str, Any] = {}
        if isinstance(blob, dict) and "model_state_dict" in blob:
            state, meta = blob["model_state_dict"], blob
        elif isinstance(blob, dict) and isinstance(blob.get("state_dict"), dict):
            state, meta = blob["state_dict"], blob
        elif isinstance(blob, dict) and any(hasattr(v, "shape") for v in blob.values()):
            state = blob  # a bare state_dict
        else:
            raise ValueError(f"Can't read a Swin state_dict out of {Path(path).name} ({type(blob).__name__}).")

        class_names = _int_keyed(meta.get("class_names") or _DEFAULT_CLASS_NAMES)
        image_size = int(meta.get("image_size", 224))
        mean = tuple(meta.get("imagenet_mean", _IMAGENET_MEAN))
        std = tuple(meta.get("imagenet_std", _IMAGENET_STD))

        model = SwinTransformer(img_size=image_size, num_classes=len(class_names),
                                norm_layer=nn.LayerNorm, **_SWIN_TINY)
        state = {k[len("module."):] if k.startswith("module.") else k: v
                 for k, v in state.items() if hasattr(v, "shape")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        # buffers (relative_position_index / attn_mask) are regenerated on build,
        # so ignore them when judging the match.
        real_missing = [k for k in missing if not k.endswith(("relative_position_index", "attn_mask"))]
        if len(real_missing) > 0.1 * len(model.state_dict()):
            raise ValueError(
                f"{Path(path).name}: {len(real_missing)} weights didn't match the Swin-Tiny "
                "architecture — is this really a Swin-Tiny checkpoint?"
            )
        if real_missing or unexpected:
            print(f"[swin] loaded {Path(path).name}: {len(real_missing)} missing, {len(unexpected)} unexpected keys")

        return cls(model, class_names=class_names, image_size=image_size, mean=mean, std=std,
                   device=device, positive_class=positive_class)

    @classmethod
    def use_default(
        cls,
        *,
        device: str = "auto",
        checkpoint: str | Path | None = None,
        positive_class: int | str | None = None,
    ) -> "SwinDetector":
        """``checkpoint=`` takes an explicit ``.pth`` path; otherwise the one
        bundled with this package (``src/weights/best_swin_tiny_binary.pth``)."""
        if checkpoint is not None:
            path = Path(checkpoint).expanduser()
        else:
            path = locate_checkpoint(
                ("best_swin_tiny_binary.pth", "best_swin*.pth", "swin*.pth"),
                script_dir=SCRIPT_DIR, env_var="SWIN_CHECKPOINT",
            )
        if path is None or not path.is_file():
            looked = ", ".join(str(d) for d in candidate_weight_dirs(SCRIPT_DIR, env_var="SWIN_CHECKPOINT"))
            raise FileNotFoundError(
                f"Swin checkpoint (best_swin_tiny_binary.pth) not found. Pass "
                "checkpoint=<path>, set $SWIN_CHECKPOINT, or drop the file in one of: "
                f"{looked}"
            )
        return cls.from_checkpoint(path, device=device, positive_class=positive_class)

    # --- scoring -----------------------------------------------------

    def _score(self, image: Image.Image) -> float:
        torch = self._torch
        tensor = self._transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self._model(tensor).float().squeeze(0)
        if logits.ndim == 0 or logits.shape[-1] == 1:
            return float(torch.sigmoid(logits.reshape(())).item())
        return float(torch.softmax(logits, dim=-1)[self._positive_index].item())
