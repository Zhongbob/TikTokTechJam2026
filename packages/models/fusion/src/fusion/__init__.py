"""Fusion model — Community-Forensics + OpenSDI combined into one detector.

`FusionDetector` (`detector_common.CombinerDetector` subclass): ``method`` ∈
``max`` (default, threshold 0.19) / ``mean`` / ``weighted`` / ``meta``.
`FusionTrainer` (`detector_common.CombinerTrainer` subclass) fits the ``weighted``
weights and the tree-based ``meta`` classifier, and has ``compare_methods()``.
"""

from fusion.detector import FusionDetector, build_default_members, DEFAULT_MAX_THRESHOLD
from fusion.trainer import FusionTrainer
from detector_common import MetaClassifier

__all__ = [
    "FusionDetector",
    "FusionTrainer",
    "MetaClassifier",
    "build_default_members",
    "DEFAULT_MAX_THRESHOLD",
]
