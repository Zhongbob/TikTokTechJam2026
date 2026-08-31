"""DRCT AI-generated-image detector (running/inference stage).

`DRCTDetector` implements the shared `detector_common.ImageDetector` /
`shared_types.interfaces.EnsembleDetector` contract. DRCT (Diffusion
Reconstruction Contrastive Training, beibuwandeluori/DRCT) trains a ConvNeXt-B
or CLIP-ViT-L/14 backbone on SD-generated images; at inference it is just
``backbone -> Linear(embedding, 2) -> real/fake``.
"""

from drct.detector import DRCTDetector
from drct._model import build_drct_model

__all__ = ["DRCTDetector", "build_drct_model"]
