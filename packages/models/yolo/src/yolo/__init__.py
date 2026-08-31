"""Baseline real-vs-AI-generated classifier: training/testing (YoloTrainer)
and running/inference (YoloDetector) stages, built on the shared
shared_types.TrainableModel / EnsembleDetector contracts.
"""

from yolo.detector import YoloDetector, load_yolo_model, resolve_ai_scorer
from yolo.trainer import YoloTrainer

__all__ = [
    "YoloDetector",
    "YoloTrainer",
    "load_yolo_model",
    "resolve_ai_scorer",
]
