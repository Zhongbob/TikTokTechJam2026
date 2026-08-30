"""DRCT AI-generated-image detector (running/inference stage).

`DRCTDetector` implements the shared `detector_common.ImageDetector` /
`shared_types.interfaces.EnsembleDetector` contract. DRCT (Diffusion
Reconstruction Contrastive Training) trains a CLIP-ViT-B/16 or ConvNeXt-Base
backbone with a reconstruction-contrastive objective; at inference it is just
``backbone -> linear head -> real/fake``.
"""

from drct.detector import DRCTDetector

__all__ = ["DRCTDetector"]
