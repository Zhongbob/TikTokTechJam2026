"""Assembles PSCC-Net's three released sub-networks into one ``nn.Module``.

PSCC-Net (``proteus1991/PSCC-Net``) ships **three** separate checkpoints, each
saved from an ``nn.DataParallel`` wrapper (so every key is ``module.*``):

    HRNet.pth          -> FENet   (HRNet-W18-small feature extractor)
    NLCDetection.pth   -> SegNet  (non-local progressive localization head)
    DetectionHead.pth  -> ClsNet  (binary "manipulated vs authentic" head)

``test.py`` runs them as::

    feat   = FENet(image)                 # [s1, s2, s3, s4]
    mask   = SegNet(feat)[0]              # B x 1 x h x w, sigmoid already applied
    mask   = F.interpolate(mask, image HxW)
    logit  = ClsNet(feat)                # B x 2
    p_forged = softmax(logit)[:, 1]

``PSCCNet.forward`` reproduces exactly that and returns a dict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from pscc_net._heads import DetectionHead, NLCDetection
from pscc_net._hrnet import get_hrnet_cfg, get_seg_model

DEFAULT_CROP_SIZE = (256, 256)


class PSCCNet(nn.Module):
    """FENet + SegNet + ClsNet in one module.

    ``crop_size`` is the grid the heads resize HRNet features to; keep it equal
    to the detector's input size so the resize is a no-op (PSCC-Net trains at
    256x256).
    """

    def __init__(self, crop_size: tuple[int, int] = DEFAULT_CROP_SIZE) -> None:
        super().__init__()
        args = {"crop_size": [int(crop_size[0]), int(crop_size[1])]}
        self.FENet = get_seg_model(get_hrnet_cfg())
        self.NLCDetection = NLCDetection(args)
        self.DetectionHead = DetectionHead(args)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.FENet(x)

        mask = self.NLCDetection(feat)[0]
        mask = F.interpolate(mask, size=(x.size(2), x.size(3)),
                             mode="bilinear", align_corners=True)

        logit = self.DetectionHead(feat)
        prob = torch.softmax(logit, dim=1)

        return {
            "mask": mask,                 # B x 1 x H x W, [0, 1]
            "logit": logit,               # B x 2
            "prob_forged": prob[:, 1],    # B, p(manipulated)
        }


# --- checkpoint loading -------------------------------------------------------

_SUBNETS = ("FENet", "NLCDetection", "DetectionHead")


def _strip_module(state: dict[str, Any]) -> dict[str, Any]:
    return {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state.items()
        if hasattr(v, "shape")
    }


def _load_one(sub: nn.Module, path: str | Path, tag: str) -> None:
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    state = blob
    if isinstance(blob, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if isinstance(blob.get(key), dict):
                state = blob[key]
                break
    state = _strip_module(state)
    missing, unexpected = sub.load_state_dict(state, strict=False)
    target = len(sub.state_dict())
    print(f"[pscc-net] {tag}: loaded {Path(path).name} — "
          f"{len(missing)}/{target} missing, {len(unexpected)} unexpected")
    if missing and len(missing) > 0.1 * max(1, target):
        print(f"[pscc-net] WARNING: {tag} has many unfilled weights — wrong checkpoint?")


def load_pscc_weights(
    model: PSCCNet,
    hrnet_path: str | Path,
    nlc_path: str | Path,
    cls_path: str | Path,
) -> PSCCNet:
    """Load the three released PSCC-Net checkpoints into ``model`` in place."""
    _load_one(model.FENet, hrnet_path, "FENet")
    _load_one(model.NLCDetection, nlc_path, "NLCDetection")
    _load_one(model.DetectionHead, cls_path, "DetectionHead")
    return model


def build_pscc_net(
    hrnet_path: str | Path,
    nlc_path: str | Path,
    cls_path: str | Path,
    *,
    crop_size: tuple[int, int] = DEFAULT_CROP_SIZE,
) -> PSCCNet:
    """Build a PSCC-Net and load the three checkpoints."""
    model = PSCCNet(crop_size=crop_size)
    load_pscc_weights(model, hrnet_path, nlc_path, cls_path)
    return model
