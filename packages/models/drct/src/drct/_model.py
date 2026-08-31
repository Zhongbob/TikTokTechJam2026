"""Minimal rebuild of DRCT's ``ContrastiveModels`` (network/models.py in
beibuwandeluori/DRCT) — just enough to load a released checkpoint for
inference, without cloning the repo.

The released DRCT-2M checkpoints are ``ContrastiveModels`` instances:

    backbone (timm ConvNeXt-B or OpenAI CLIP-ViT) -> Linear(., embedding_size)
    -> self.fc = Linear(embedding_size, num_classes=2)

so the state dict keys are ``model.*`` (backbone) + ``fc.*`` (classifier).
"""

from __future__ import annotations

import torch.nn as nn

#: DRCT trains/evaluates every backbone with ImageNet normalisation
#: (data/transform.py create_val_transforms), not CLIP normalisation.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_CLIP_CHANNELS = {"ViT-B/32": 512, "ViT-B/16": 512, "ViT-L/14": 768, "RN50": 1024}


class ContrastiveModel(nn.Module):
    """``backbone -> Linear(embedding_size, num_classes)``. Keys match DRCT's
    ``ContrastiveModels`` (``model.*`` + ``fc.*``)."""

    def __init__(self, backbone: nn.Module, embedding_size: int = 1024, num_classes: int = 2) -> None:
        super().__init__()
        self.model = backbone
        self.fc = nn.Linear(embedding_size, num_classes)

    def forward(self, x):  # noqa: D401
        return self.fc(self.model(x))


class _CLIPVisualBackbone(nn.Module):
    """DRCT's ``CLIPModelV2``: OpenAI CLIP image encoder -> Linear(embedding_size)."""

    def __init__(self, clip_name: str, embedding_size: int) -> None:
        super().__init__()
        try:
            import clip
        except ImportError as error:  # pragma: no cover
            raise ImportError(
                "The DRCT CLIP-ViT variant needs OpenAI clip: "
                "pip install git+https://github.com/openai/CLIP.git"
            ) from error
        self.model, _ = clip.load(clip_name, device="cpu")
        self.model.float()
        self.fc = nn.Linear(_CLIP_CHANNELS[clip_name], embedding_size)

    def forward(self, x):
        return self.fc(self.model.encode_image(x))


def build_drct_model(
    model_name: str = "convnext_base_in22k",
    *,
    embedding_size: int = 1024,
    num_classes: int = 2,
) -> ContrastiveModel:
    """Build an untrained DRCT model matching a released checkpoint.

    ``model_name`` accepts ``convnext_base_in22k`` (default) / any timm
    ``convnext_*`` name, or ``clip-ViT-L-14`` / ``clip-ViT-B-16``.
    """
    if "convnext" in model_name:
        import timm

        net = timm.create_model(model_name, pretrained=False)
        net.head.fc = nn.Linear(net.head.fc.in_features, embedding_size)
        backbone: nn.Module = net
    elif "clip" in model_name.lower():
        clip_name = model_name.replace("clip-", "").replace("L-", "L/").replace("B-", "B/")
        if clip_name not in _CLIP_CHANNELS:
            raise ValueError(f"Unknown CLIP variant {clip_name!r}; expected one of {list(_CLIP_CHANNELS)}")
        backbone = _CLIPVisualBackbone(clip_name, embedding_size)
    else:
        raise ValueError(f"Unsupported DRCT model_name {model_name!r} (expected convnext_* or clip-ViT-*).")

    return ContrastiveModel(backbone, embedding_size=embedding_size, num_classes=num_classes)
