"""Inference wrapper around a DRCT detector.

DRCT ("Diffusion Reconstruction Contrastive Training") releases several
checkpoints — CLIP-ViT-B/16 and ConvNeXt-Base backbones, each trained on
SD v1.4 and SD v2.0 separately. The reconstruction/contrastive machinery is
train-time only; inference is::

    features = backbone.encode_image(preprocess(image))
    logits   = head(features)              # 2 logits: real / fake
    p_fake   = softmax(logits)[fake_index]

`torch` / `open_clip` are imported lazily.

⚠️ FIDELITY: the checkpoint layout (whether it stores a fine-tuned backbone +
head, or just a head on frozen CLIP features) varies. `from_checkpoint()` tries
to load a `visual.*` / `head.*` split and falls back to head-only. Confirm
`positive_index` and the backbone/pretrained tag against the checkpoint you use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from detector_common import ImageDetector, resolve_device
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR.parent / "weights" / "drct_clip_vit_b16_sdv14.pth"

#: (open_clip model name, pretrained tag) for the default backbone.
DEFAULT_BACKBONE = ("ViT-B-16", "openai")
_FEATURE_DIMS = {"ViT-B-16": 512, "ViT-L-14": 768, "convnext_base_w": 640}


class DRCTDetector(ImageDetector):
    """DRCT backbone + linear head, wrapped as an `ImageDetector`.

        detector = DRCTDetector.use_default()
        result = detector.predict(pil_image)
    """

    name = "drct-clip-vit-b16"
    is_placeholder = False

    def __init__(
        self,
        backbone: Any,
        head: Any,
        preprocess: Any,
        *,
        device: str = "auto",
        positive_index: int = 1,
    ) -> None:
        import torch

        self.device = resolve_device(device)
        self._backbone = backbone.to(self.device).eval()
        self._head = head.to(self.device).eval()
        self._preprocess = preprocess
        self._torch = torch
        self._positive_index = positive_index

    # --- construction --------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "auto",
        backbone: str = DEFAULT_BACKBONE[0],
        pretrained: str = DEFAULT_BACKBONE[1],
        num_classes: int = 2,
        positive_index: int = 1,
    ) -> "DRCTDetector":
        import open_clip
        import torch
        from torch import nn

        model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
        visual = model.visual
        feat_dim = _FEATURE_DIMS.get(backbone, getattr(model, "output_dim", 512))
        head = nn.Linear(feat_dim, num_classes)

        state = torch.load(str(path), map_location="cpu")
        state = state.get("model", state.get("state_dict", state))
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

        visual_sd = {k[len("visual."):]: v for k, v in state.items() if k.startswith("visual.")}
        head_sd = {k[len("head."):]: v for k, v in state.items()
                   if k.startswith("head.") or k.startswith("fc.") or k.startswith("classifier.")}
        head_sd = {k.split(".", 1)[-1] if k.startswith(("fc.", "classifier.")) else k: v
                   for k, v in head_sd.items()}
        if visual_sd:
            visual.load_state_dict(visual_sd, strict=False)
        if head_sd:
            head.load_state_dict(head_sd, strict=False)
        elif not visual_sd:
            # whole checkpoint is the head
            head.load_state_dict(state, strict=False)

        return cls(visual, head, preprocess, device=device, positive_index=positive_index)

    @classmethod
    def use_default(cls, *, device: str = "auto") -> "DRCTDetector":
        if not DEFAULT_CHECKPOINT.is_file():
            raise FileNotFoundError(
                f"DRCT checkpoint not found at {DEFAULT_CHECKPOINT}. Download one of the DRCT "
                "checkpoints (CLIP-ViT-B/16 or ConvNeXt-Base, SDv1.4 / SDv2.0) from the official "
                "release and place it there, or call from_checkpoint(path, backbone=..., pretrained=...)."
            )
        return cls.from_checkpoint(DEFAULT_CHECKPOINT, device=device)

    # --- scoring ---------------------------------------------------

    def _score(self, image: Image.Image) -> float:
        torch = self._torch
        tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self._backbone(tensor)
            if isinstance(features, (tuple, list)):
                features = features[0]
            logits = self._head(features.float()).squeeze(0)
        if logits.ndim == 0 or logits.shape[-1] == 1:
            return float(torch.sigmoid(logits.reshape(())).item())
        return float(torch.softmax(logits, dim=-1)[self._positive_index].item())
