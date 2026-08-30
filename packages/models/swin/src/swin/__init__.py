"""Swin-Tiny real-vs-AI-generated image classifier (running/inference stage).

`SwinDetector` implements the shared `detector_common.ImageDetector` /
`shared_types.interfaces.EnsembleDetector` contract, loading the ``.pth``
checkpoint produced by ``packages/models/swin/s3_swin_transformer_streaming.ipynb``
(a fine-tuned ``microsoft/Swin-Transformer`` Swin-Tiny).
"""

from swin.detector import SwinDetector

__all__ = ["SwinDetector"]
