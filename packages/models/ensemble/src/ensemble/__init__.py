"""Ensemble model — the fusion model + the standalone trained classifiers.

`EnsembleDetector` (`detector_common.CombinerDetector` subclass): members are
``fusion`` (CF + OpenSDI), ``convnext_aigc``, ``clip_vit_b32``, ``dinov2``,
``normal_classifier`` and ``swin``; ``method`` ∈ ``max`` / ``mean`` /
``weighted`` / ``meta``. ``use_autoencoder=True`` restores the image before the
fusion member only.

`EnsembleTrainer` (`detector_common.CombinerTrainer` subclass) fits the
``weighted`` weights and the tree-based ``meta`` classifier.
"""

from ensemble.detector import (
    DEFAULT_MEMBERS,
    EnsembleDetector,
    build_default_ensemble_members,
)
from ensemble.trainer import EnsembleTrainer

__all__ = [
    "EnsembleDetector",
    "EnsembleTrainer",
    "build_default_ensemble_members",
    "DEFAULT_MEMBERS",
]
