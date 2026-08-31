"""Trainer for the fusion model's (optional, future) meta-classifier.

Right now `FusionDetector` combines its members with a fixed rule
(``combine="max"``) and needs **no training**. This module is a placeholder for
the day a learned combiner is wanted instead: a small classifier (logistic
regression / gradient-boosted trees / a 2-layer MLP) that takes the vector of
member p(ai) scores — e.g. ``[p_community_forensics, p_pscc_net]`` — plus
optionally cheap extra features (PSCC-Net mask stats, image size, …) and outputs
the fused p(ai).

Every method below is intentionally a stub that raises ``NotImplementedError``.
Wiring order when we do build it:

    1. `_extract_features(samples)` — run each member once per image, stack the
       score vectors, return (X, y).
    2. `train(samples)` — fit the meta-classifier on (X, y); store it.
    3. `save(path)` / `load(path)` — persist just the meta-classifier
       (the members are reconstructed from their own packages).
    4. `FusionTrainer.load(path).as_detector()` -> `FusionDetector` with
       ``combine="meta"`` via `FusionDetector.attach_meta_classifier`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from shared_types.training import (
    ClassifierTrainableModel,
    LabeledImageSample,
    TrainingResult,
)

_STUB = (
    "FusionTrainer is a placeholder — the fusion model currently uses the fixed "
    "'max' decider and needs no training. Implement this only if a learned "
    "meta-classifier is required."
)


class FusionTrainer(ClassifierTrainableModel):
    """Fits a meta-classifier over member detector scores. **Not implemented.**"""

    name = "fusion-meta-trainer"

    def __init__(
        self,
        members: Sequence[Any] | None = None,
        *,
        meta_classifier: Any | None = None,
    ) -> None:
        # members: the same detector objects FusionDetector would hold, used to
        # turn images into score vectors. meta_classifier: an unfitted estimator
        # (e.g. sklearn LogisticRegression) — injected so this package doesn't
        # hard-depend on a specific ML lib.
        self._members = list(members) if members is not None else []
        self._meta = meta_classifier

    # --- feature extraction ------------------------------------------

    def _extract_features(
        self, samples: Iterable[LabeledImageSample]
    ) -> tuple[list[list[float]], list[int]]:
        """images -> (X = per-image member score vectors, y = binary labels).

        Planned: for each sample, ``[m.predict(sample.image).ai_generated_probability
        for m in self._members]`` as the row, ``sample.metadata["binary_aigc_label"]``
        as the target.
        """
        raise NotImplementedError(_STUB)

    # --- TrainableModel surface -------------------------------------

    def train(
        self, samples: Iterable[LabeledImageSample], **kwargs: Any
    ) -> TrainingResult:
        """Fit the meta-classifier on member score vectors. **Not implemented.**"""
        raise NotImplementedError(_STUB)

    def evaluate(
        self, samples: Iterable[LabeledImageSample], **kwargs: Any
    ) -> dict[str, float]:
        """Score the fitted meta-classifier on held-out samples. **Not implemented.**"""
        raise NotImplementedError(_STUB)

    def save(self, path: str | Path) -> None:
        """Persist the fitted meta-classifier (members are not saved — they come
        from their own packages). **Not implemented.**"""
        raise NotImplementedError(_STUB)

    @classmethod
    def load(cls, path: str | Path) -> "FusionTrainer":
        """Reconstruct a fitted `FusionTrainer` from `save()`. **Not implemented.**"""
        raise NotImplementedError(_STUB)

    # --- handoff to inference --------------------------------------

    def as_detector(self, **fusion_kwargs: Any) -> Any:
        """Build a `FusionDetector` that uses this trained meta-classifier
        (``combine="meta"``). **Not implemented.**"""
        raise NotImplementedError(_STUB)
