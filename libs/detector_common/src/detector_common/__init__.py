"""Shared plumbing for AI-generated-image detectors.

* `ImageDetector` — base for a single-model detector; subclass implements
  ``_score(image) -> p(ai_generated)`` and gets ``predict`` / ``evaluate``.
* `CombinerDetector` / `CombinerTrainer` — base for a detector that fuses several
  member detectors (``fusion``, ``ensemble``): ``method`` ∈ max / mean / weighted
  / meta, with a grid-search weight fitter and a tree-based `MetaClassifier`.
"""

from detector_common.base import ImageDetector, resolve_device, save_confusion_matrix
from detector_common.combiner import CombinerDetector, METHODS, meta_score
from detector_common.combiner_trainer import CombinerTrainer
from detector_common.meta import MetaClassifier, default_estimator

__all__ = [
    "ImageDetector",
    "resolve_device",
    "save_confusion_matrix",
    "CombinerDetector",
    "CombinerTrainer",
    "MetaClassifier",
    "default_estimator",
    "meta_score",
    "METHODS",
]
