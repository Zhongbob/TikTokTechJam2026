"""In-house fine-tuned CLIP ViT-B/32 real-vs-AI-generated image detector (inference).

`ClipViTB32Detector` implements the shared `detector_common.ImageDetector` /
`shared_types.interfaces.EnsembleDetector` contract, loading a checkpoint that
fine-tuned the last visual blocks of OpenAI CLIP ViT-B/32 and classifies by
text-image similarity against per-class prompt sets.
"""

from clip_vit_b32.detector import ClipViTB32Detector

__all__ = ["ClipViTB32Detector"]
