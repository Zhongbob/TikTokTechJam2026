"""In-house fine-tuned ConvNeXt real-vs-AI-generated image detector (inference).

`ConvNextAIGCDetector` implements the shared `detector_common.ImageDetector` /
`shared_types.interfaces.EnsembleDetector` contract, loading a Hugging Face
`ConvNextForImageClassification` checkpoint (``convnext_aigc_run*.zip`` at the
repo root, or an extracted model dir).
"""

from convnext_aigc.detector import ConvNextAIGCDetector

__all__ = ["ConvNextAIGCDetector"]
