"""Inference wrapper around a DRCT detector (beibuwandeluori/DRCT).

DRCT ("Diffusion Reconstruction Contrastive Training") releases checkpoints
trained on SD-generated images. The reconstruction/contrastive objective is
train-time only; inference is just::

    logits = model(preprocess(image))     # ContrastiveModel -> [B, 2]
    p_fake = softmax(logits)[:, 1]

The architecture (``ContrastiveModel``) is rebuilt from the vendored
``_model.py`` — no repo clone needed for the ConvNeXt variant (needs ``timm``);
the CLIP-ViT variant additionally needs OpenAI ``clip``.

Recommended checkpoint (ModelScope ``BokingChen/DRCT-2M`` -> ``pretrained.zip``):
``convnext_base_in22k_224_drct_amp_crop/14_acc0.9996.pth`` (ConvNeXt-B, SD v1.4).

Preprocessing matches DRCT's ``create_val_transforms(size=224, is_crop=True)``:
centre-crop 224 (pad if smaller) + ImageNet normalisation.

`torch` / `timm` are imported lazily.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable

from detector_common import ImageDetector, resolve_device
from PIL import Image

from drct._model import IMAGENET_MEAN, IMAGENET_STD, build_drct_model

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR.parent / "weights" / "drct_convnext_base_sdv14.pth"

DEFAULT_MODEL_NAME = "convnext_base_in22k"
DEFAULT_EMBEDDING_SIZE = 1024
DEFAULT_INPUT_SIZE = 224


class DRCTDetector(ImageDetector):
    """DRCT ConvNeXt-B / CLIP-ViT + linear head, wrapped as an `ImageDetector`.

        detector = DRCTDetector.from_checkpoint("14_acc0.9996.pth")   # convnext, default
        detector = DRCTDetector.from_checkpoint(
            "13_acc0.9664.pth", model_name="clip-ViT-L-14"
        )
        result = detector.predict(pil_image)
        metrics = detector.evaluate(val_samples, generate_confusion_matrix=True)

    ``positive_index`` (default 1) is the "fake" logit; ``flip=True`` inverts the
    final score if a labelled eval comes out reversed.
    """

    name = "drct-convnext-base"
    is_placeholder = False

    def __init__(
        self,
        model: Any,
        *,
        device: str = "auto",
        image_size: int = DEFAULT_INPUT_SIZE,
        mean: tuple[float, float, float] = IMAGENET_MEAN,
        std: tuple[float, float, float] = IMAGENET_STD,
        positive_index: int = 1,
        flip: bool = False,
        name: str | None = None,
    ) -> None:
        import torch
        from torchvision import transforms

        self.device = resolve_device(device)
        self._torch = torch
        self._model = model.to(self.device).eval()
        self._positive_index = positive_index
        self.flip = flip
        if name:
            self.name = name
        # DRCT --is_crop: take the centre 224 crop (CenterCrop pads with 0 first
        # if the image is smaller), then ImageNet-normalise.
        self._transform = transforms.Compose([
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(list(mean), list(std)),
        ])

    # --- construction --------------------------------------------------

    @classmethod
    def from_module(cls, model: Any, **kwargs: Any) -> "DRCTDetector":
        """Wrap an already-built + already-loaded DRCT ``ContrastiveModel``."""
        return cls(model, **kwargs)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "auto",
        model_name: str = DEFAULT_MODEL_NAME,
        embedding_size: int = DEFAULT_EMBEDDING_SIZE,
        image_size: int = DEFAULT_INPUT_SIZE,
        positive_index: int = 1,
        flip: bool = False,
        arch_factory: Callable[[], Any] | None = None,
    ) -> "DRCTDetector":
        import torch

        model = arch_factory() if arch_factory is not None else build_drct_model(
            model_name, embedding_size=embedding_size, num_classes=2
        )

        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        state: Any = blob
        if isinstance(blob, dict):
            for key in ("model_state_dict", "state_dict", "model", "net"):
                if isinstance(blob.get(key), dict):
                    state = blob[key]
                    break
        state = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state.items() if hasattr(v, "shape")
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        target = len(model.state_dict())
        print(f"[drct] loaded {Path(path).name}: {len(missing)}/{target} missing, {len(unexpected)} unexpected keys")
        if len(missing) > 0.3 * max(1, target):
            print(
                "[drct] WARNING: many weights did not match — check model_name "
                f"({model_name!r}) / embedding_size ({embedding_size})."
            )

        display_name = "drct-clip" if "clip" in model_name.lower() else "drct-convnext-base"
        return cls(
            model, device=device, image_size=image_size,
            positive_index=positive_index, flip=flip, name=display_name,
        )

    @classmethod
    def use_default(cls, *, device: str = "auto") -> "DRCTDetector":
        if not DEFAULT_CHECKPOINT.is_file():
            raise FileNotFoundError(
                f"DRCT checkpoint not found at {DEFAULT_CHECKPOINT}. Download DRCT-2M's "
                "pretrained.zip from https://modelscope.cn/datasets/BokingChen/DRCT-2M/files , "
                "put convnext_base_in22k_224_drct_amp_crop/14_acc0.9996.pth there (renamed to "
                f"{DEFAULT_CHECKPOINT.name}), or call from_checkpoint(path, model_name=...)."
            )
        return cls.from_checkpoint(DEFAULT_CHECKPOINT, device=device)

    # --- scoring -----------------------------------------------------

    def raw_output(self, image: Image.Image) -> Any:
        """The model's raw logits for one image."""
        torch = self._torch
        tensor = self._transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self._model(tensor)

    def _score(self, image: Image.Image) -> float:
        torch = self._torch
        logits = self.raw_output(image).reshape(-1).float()
        if logits.numel() >= 2:
            p_ai = float(torch.softmax(logits, dim=-1)[self._positive_index].item())
        else:
            p_ai = 1.0 / (1.0 + math.exp(-float(logits[0].item())))
        p_ai = min(1.0, max(0.0, p_ai))
        return 1.0 - p_ai if self.flip else p_ai
