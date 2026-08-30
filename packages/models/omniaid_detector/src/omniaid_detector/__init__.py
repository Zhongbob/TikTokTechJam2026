"""OmniAID AI-generated-image detector (running/inference stage).

`OmniAIDDetector` implements the shared `detector_common.ImageDetector` /
`shared_types.interfaces.EnsembleDetector` contract, wrapping OmniAID
(yunncheng/OmniAID) — a Mixture-of-Experts detector over a DINOv3 ViT-L/16 or
CLIP-ViT-L/14 backbone.
"""

from omniaid_detector.detector import OmniAIDDetector

__all__ = ["OmniAIDDetector"]
