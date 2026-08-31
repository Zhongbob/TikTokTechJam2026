"""Fusion model — Community-Forensics + OpenSDI combined into one detector.

`FusionDetector` (inference) implements the shared
`detector_common.ImageDetector` / `shared_types.interfaces.EnsembleDetector`
contract. ``method`` picks how member p(ai) scores are fused before the
threshold:

* ``"max"`` / ``"threshold"`` (default) — the simple threshold split.
* ``"weighted"`` — ``sum(w_i * p_i)``; fit ``weights`` with `FusionTrainer`.
* ``"meta"`` — a learned combiner (still a trainer stub).

`FusionTrainer` (training) implements ``optimal_weights()`` — a grid search over
the member-weight simplex for the split that best separates real from AI on
example data — plus ``train`` / ``evaluate`` / ``save`` / ``load`` /
``as_detector``. The meta-classifier path is still a stub.
"""

from fusion.detector import FusionDetector, build_default_members, DEFAULT_MAX_THRESHOLD
from fusion.trainer import FusionTrainer

__all__ = [
    "FusionDetector",
    "FusionTrainer",
    "build_default_members",
    "DEFAULT_MAX_THRESHOLD",
]
