"""In-house fine-tuned DINOv2 real-vs-AI-generated image detector (inference).

`DINOv2Detector` implements the shared `detector_common.ImageDetector` /
`shared_types.interfaces.EnsembleDetector` contract, loading a checkpoint with a
HF `Dinov2Model` backbone (``facebook/dinov2-small`` by default) + a linear head
on the CLS token.
"""

from dinov2.detector import DINOv2Detector

__all__ = ["DINOv2Detector"]
