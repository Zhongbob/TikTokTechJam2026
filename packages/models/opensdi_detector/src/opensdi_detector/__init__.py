"""OpenSDI / MaskCLIP manipulation detector (running/inference stage).

`OpenSDIDetector` implements the shared `detector_common.ImageDetector` /
`shared_types.interfaces.EnsembleDetector` contract, wrapping MaskCLIP from
iamwangyabin/OpenSDI — a frozen CLIP-ViT-L/14 + MAE-style side-adapter that
outputs both an image-level real/manipulated label and a pixel-level forgery
mask. The wrapper turns either into a single ``p(ai_generated)`` score.

`setup_opensdi()` (in `opensdi_detector.bootstrap`) provisions the repo clone,
pip deps and weights the detector needs — run it once before `use_default()`.
"""

from opensdi_detector.detector import OpenSDIDetector
from opensdi_detector.bootstrap import setup_opensdi

__all__ = ["OpenSDIDetector", "setup_opensdi"]
