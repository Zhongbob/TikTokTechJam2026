"""Fusion model — Community-Forensics + OpenSDI combined into one detector.

`FusionDetector` (inference) implements the shared
`detector_common.ImageDetector` / `shared_types.interfaces.EnsembleDetector`
contract. ``method`` picks how member p(ai) scores are fused before the
threshold:

* ``"max"`` / ``"threshold"`` (default, threshold 0.19) — the simple OR-rule.
* ``"weighted"`` — ``sum(w_i * p_i)``; fit ``weights`` with `FusionTrainer`.
* ``"meta"`` — a tree-based combiner (`fusion._meta.MetaClassifier`) fitted by
  `FusionTrainer.fit_meta_classifier`.

`FusionTrainer` (training) implements ``optimal_weights()`` (weighted method),
``fit_meta_classifier()`` (meta method — dependency-free CART / sklearn / xgboost)
and ``compare_methods()`` to benchmark max / weighted / meta side by side.
"""

from fusion.detector import FusionDetector, build_default_members, DEFAULT_MAX_THRESHOLD
from fusion.trainer import FusionTrainer
from fusion._meta import MetaClassifier

__all__ = [
    "FusionDetector",
    "FusionTrainer",
    "MetaClassifier",
    "build_default_members",
    "DEFAULT_MAX_THRESHOLD",
]
