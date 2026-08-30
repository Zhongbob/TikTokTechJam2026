"""Inference wrapper around OmniAID (yunncheng/OmniAID, ICML 2026).

OmniAID = "Decoupling Semantic and Artifacts for Universal AI-Generated Image
Detection in the Wild": a hybrid Mixture-of-Experts (routable semantic experts +
one always-on artifact expert) on top of a frozen DINOv3 ViT-L/16 or
CLIP-ViT-L/14@336 backbone. It emits a single scalar per image.

The build config + preprocessing here mirror the official ``reward/clean_test.py``
(HF checkpoints under ``Yunncheng/OmniAID``):

    clip : checkpoint_omniaid_v2.pth      | CLIP-ViT-L/14@336 | res 336 | CLIP norm
    dino : checkpoint_omniaid_dino_v2.pth | DINOv3 ViT-L/16    | res 448 | ImageNet norm

`torch` / `torchvision` are imported lazily.

⚠️ FIDELITY: OmniAID's model classes (``OmniAID`` in ``omniaid.py`` /
``OmniAID_DINO`` in ``omniaid-dino.py``) are NOT vendored — they need the
official repo. Pass ``repo_dir=`` (a clone of yunncheng/OmniAID) or an
``arch_factory``. The single output scalar's sign convention isn't documented;
if scores come out inverted on a labelled set, construct with ``flip=True``.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable

from detector_common import ImageDetector, resolve_device
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_DIR = SCRIPT_DIR.parent / "weights"

#: Per-backbone presets copied from OmniAID's reward/clean_test.py.
BACKBONES: dict[str, dict[str, Any]] = {
    "clip": {
        "module": "omniaid",
        "class": "OmniAID",
        "checkpoint": "checkpoint_omniaid_v2.pth",
        "resolution": 336,
        "mean": (0.48145466, 0.4578275, 0.40821073),
        "std": (0.26862954, 0.26130258, 0.27577711),
        "config": dict(
            CLIP_path="openai/clip-vit-large-patch14-336",
            num_experts=6, rank_per_expert=1, moe_top_k=2,
            moe_router_hidden_dim=256, is_hybrid=True,
        ),
    },
    "dino": {
        "module": "omniaid-dino",
        "class": "OmniAID_DINO",
        "checkpoint": "checkpoint_omniaid_dino_v2.pth",
        "resolution": 448,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "config": dict(
            DINOV3_path="facebook/dinov3-vitl16-pretrain-lvd1689m",
            num_experts=6, rank_per_expert=1, moe_top_k=2,
            moe_router_hidden_dim=256, is_hybrid=True,
        ),
    },
}


def _import_class(module_name: str, class_name: str, repo_dir: str | os.PathLike[str] | None) -> Any:
    """Import ``class_name`` from OmniAID's ``module_name`` (which may contain a
    hyphen, e.g. ``omniaid-dino``), adding ``repo_dir`` to ``sys.path`` first."""
    if repo_dir is not None:
        resolved = str(Path(repo_dir).resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    if "-" in module_name:
        if repo_dir is None:
            raise ImportError(f"Importing '{module_name}' needs repo_dir= (the OmniAID clone).")
        file = Path(repo_dir).resolve() / f"{module_name}.py"
        spec = importlib.util.spec_from_file_location(module_name.replace("-", "_"), file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load {file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    return getattr(module, class_name)


class OmniAIDDetector(ImageDetector):
    """OmniAID MoE detector, wrapped as an `ImageDetector`.

        # with a clone of yunncheng/OmniAID on disk
        detector = OmniAIDDetector.from_checkpoint(
            "checkpoint_omniaid_dino_v2.pth", backbone="dino", repo_dir="/content/OmniAID"
        )
        result = detector.predict(pil_image)
        metrics = detector.evaluate(val_samples, generate_confusion_matrix=True)

    ``flip=True`` inverts the score if OmniAID's scalar turns out to mean
    "authentic" rather than "AI-generated" on your labelled set.
    """

    name = "omniaid"
    is_placeholder = False

    def __init__(
        self,
        model: Any,
        *,
        resolution: int = 448,
        mean: tuple[float, float, float] = BACKBONES["dino"]["mean"],
        std: tuple[float, float, float] = BACKBONES["dino"]["std"],
        device: str = "auto",
        flip: bool = False,
        output_is_probability: bool | None = None,
        name: str | None = None,
    ) -> None:
        import torch
        from torchvision import transforms

        self.device = resolve_device(device)
        self._torch = torch
        self._model = model.to(self.device).eval()
        self.flip = flip
        self._output_is_probability = output_is_probability
        if name:
            self.name = name
        self._transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(mean), std=list(std)),
        ])

    # --- construction --------------------------------------------------

    @classmethod
    def from_module(
        cls,
        model: Any,
        *,
        backbone: str = "dino",
        device: str = "auto",
        flip: bool = False,
        output_is_probability: bool | None = None,
    ) -> "OmniAIDDetector":
        """Wrap an already-built + already-loaded OmniAID ``nn.Module``."""
        spec = BACKBONES[backbone]
        return cls(
            model, resolution=spec["resolution"], mean=spec["mean"], std=spec["std"],
            device=device, flip=flip, output_is_probability=output_is_probability,
            name=f"omniaid-{backbone}",
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        backbone: str = "dino",
        repo_dir: str | os.PathLike[str] | None = None,
        arch_factory: Callable[[], Any] | None = None,
        device: str = "auto",
        flip: bool = False,
        output_is_probability: bool | None = None,
    ) -> "OmniAIDDetector":
        import types

        import torch

        if backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {list(BACKBONES)}")
        spec = BACKBONES[backbone]

        if arch_factory is not None:
            model = arch_factory()
        else:
            model_cls = _import_class(spec["module"], spec["class"], repo_dir)
            model = model_cls(types.SimpleNamespace(**spec["config"]))

        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        state = blob["model"] if isinstance(blob, dict) and "model" in blob else blob
        state = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[omniaid] loaded {Path(path).name}: {len(missing)} missing, {len(unexpected)} unexpected keys")

        return cls.from_module(
            model, backbone=backbone, device=device, flip=flip,
            output_is_probability=output_is_probability,
        )

    @classmethod
    def use_default(
        cls,
        *,
        backbone: str = "dino",
        repo_dir: str | os.PathLike[str] | None = None,
        device: str = "auto",
        flip: bool = False,
    ) -> "OmniAIDDetector":
        repo_dir = repo_dir or os.environ.get("OMNIAID_REPO")
        checkpoint = DEFAULT_WEIGHTS_DIR / BACKBONES[backbone]["checkpoint"]
        if not checkpoint.is_file() or not repo_dir:
            raise FileNotFoundError(
                "OmniAID needs a checkpoint and the official repo. Clone "
                "https://github.com/yunncheng/OmniAID, download a checkpoint from "
                f"HF 'Yunncheng/OmniAID' to {checkpoint}, then either set OMNIAID_REPO=<clone> "
                "or call from_checkpoint(path, backbone=..., repo_dir=<clone>)."
            )
        return cls.from_checkpoint(checkpoint, backbone=backbone, repo_dir=repo_dir, device=device, flip=flip)

    # --- scoring -----------------------------------------------------

    def _score(self, image: Image.Image) -> float:
        torch = self._torch
        tensor = self._transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self._model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        value = float(output.reshape(-1)[0].item())

        is_probability = self._output_is_probability
        if is_probability is None:
            is_probability = 0.0 <= value <= 1.0
        p_ai = value if is_probability else 1.0 / (1.0 + math.exp(-value))
        return 1.0 - p_ai if self.flip else p_ai
