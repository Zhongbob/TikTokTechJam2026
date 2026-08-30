"""Baseline real-vs-AI-generated classifier: training/testing (NormalClassifierTrainer)
and running/inference (NormalClassifierDetector) stages, built on the shared
shared_types.TrainableModel / EnsembleDetector contracts.
"""

from normal_classifier.detector import NormalClassifierDetector, resolve_ai_scorer
from normal_classifier.trainer import NormalClassifierTrainer

__all__ = ["NormalClassifierDetector", "NormalClassifierTrainer", "resolve_ai_scorer"]
