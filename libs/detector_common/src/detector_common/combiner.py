"""`CombinerDetector` — the shared engine for detectors that fuse several member
detectors into one verdict (``fusion.FusionDetector``, ``ensemble.EnsembleDetector``).

A member is anything with ``.name`` and ``.predict(image) -> DetectionResult``
(so a member can itself be a `CombinerDetector`, or a plain `ImageDetector`, or
a thin adapter). ``method`` selects how the member ``p(ai)`` vector is fused into
one score, which is then compared to ``decision_threshold``:

* ``"max"`` / ``"threshold"`` — ``max(p_i)``; fake if *any* member fires.
* ``"mean"``     — unweighted average.
* ``"weighted"`` — ``sum(w_i p_i) / sum(w_i)``; needs ``weights=`` (fit with the
  matching ``CombinerTrainer``).
* ``"meta"``     — a trained `detector_common.meta.MetaClassifier` (or any object
  with ``predict_fake_proba`` / ``predict_proba``) over the raw member vector.

Subclasses set ``name`` / ``name_prefix`` and, if they have a tuned operating
point, ``default_max_threshold``.
"""

from __future__ import annotations

from typing import Any, Sequence

from PIL import Image

from detector_common.base import ImageDetector
from shared_types.detection import DetectionResult, EnsembleMemberResult

#: Public methods. ``"threshold"`` is an alias for ``"max"``.
METHODS = {"max", "mean", "weighted", "meta"}
METHOD_ALIASES = {"threshold": "max"}


def meta_score(meta: Any, rows: list[list[float]]) -> list[float]:
    """p(ai) per row from a meta classifier — `MetaClassifier.predict_fake_proba`
    if present, else the positive column of a sklearn-style ``predict_proba``."""
    if hasattr(meta, "predict_fake_proba"):
        return [float(v) for v in meta.predict_fake_proba(rows)]
    out = []
    for proba in meta.predict_proba(rows):
        out.append(float(proba[1] if len(proba) > 1 else proba[0]))
    return out


class CombinerDetector(ImageDetector):
    """Combine member detectors' ``p(ai)`` into one score + verdict."""

    name = "combiner-max"
    is_placeholder = False
    #: used to name derived variants: ``f"{name_prefix}-{method}"``
    name_prefix = "combiner"
    #: ``decision_threshold`` default for ``method="max"`` (other methods -> 0.5)
    default_max_threshold: float = 0.5

    def __init__(
        self,
        members: Sequence[Any],
        *,
        method: str = "max",
        weights: Sequence[float] | None = None,
        decision_threshold: float | None = None,
        meta_classifier: Any | None = None,
        name: str | None = None,
    ) -> None:
        if not members:
            raise ValueError(f"{type(self).__name__} needs at least one member detector")

        method = METHOD_ALIASES.get(method, method)
        if method not in METHODS:
            raise ValueError(f"method must be one of {sorted(METHODS)} (or 'threshold' == 'max')")
        if decision_threshold is None:
            decision_threshold = self.default_max_threshold if method == "max" else 0.5
        if method == "weighted" and (weights is None or len(weights) != len(members)):
            raise ValueError("method='weighted' needs weights= matching the members")
        if method == "meta" and meta_classifier is None:
            raise ValueError(
                "method='meta' needs a trained meta_classifier — fit one with the "
                "matching CombinerTrainer (or use method='max' / 'weighted')."
            )

        self._members = list(members)
        self.method = method
        self._weights = [float(w) for w in weights] if weights is not None else None
        self.decision_threshold = decision_threshold
        self._meta = meta_classifier
        self.name = name or f"{self.name_prefix}-{method}"

    # --- construction --------------------------------------------------

    @classmethod
    def from_members(cls, members: Sequence[Any], **kwargs: Any) -> "CombinerDetector":
        """Wrap an explicit list of already-constructed member detectors."""
        return cls(members, **kwargs)

    # --- scoring -----------------------------------------------------

    @property
    def members(self) -> list[Any]:
        return list(self._members)

    @property
    def weights(self) -> list[float] | None:
        return list(self._weights) if self._weights is not None else None

    def member_predictions(self, image: Image.Image) -> list[DetectionResult]:
        """Run every member on one image, in ``self.members`` order."""
        rgb = image.convert("RGB")
        return [member.predict(rgb) for member in self._members]

    def fuse_scores(self, probs: Sequence[float]) -> float:
        """Combine one member ``p(ai)`` vector into a single score."""
        if self.method == "max":
            return max(probs)
        if self.method == "mean":
            return sum(probs) / len(probs)
        if self.method == "weighted":
            assert self._weights is not None
            total = sum(self._weights) or 1.0
            return sum(p * w for p, w in zip(probs, self._weights)) / total
        return meta_score(self._meta, [list(probs)])[0]

    def _score(self, image: Image.Image) -> float:
        probs = [r.ai_generated_probability for r in self.member_predictions(image)]
        return min(1.0, max(0.0, self.fuse_scores(probs)))

    def predict(self, image: Image.Image) -> DetectionResult:
        results = self.member_predictions(image)
        probs = [r.ai_generated_probability for r in results]

        member_results = tuple(
            EnsembleMemberResult(
                model_name=member.name,
                ai_generated_probability=r.ai_generated_probability,
                confidence=(
                    r.member_results[0].confidence
                    if r.member_results
                    else self._confidence(r.ai_generated_probability)
                ),
                is_placeholder=bool(getattr(member, "is_placeholder", False)),
            )
            for member, r in zip(self._members, results)
        )

        p_ai = min(1.0, max(0.0, self.fuse_scores(probs)))
        return DetectionResult(
            verdict="ai_generated" if p_ai >= self.decision_threshold else "real",
            ai_generated_probability=p_ai,
            member_results=member_results,
            is_placeholder=self.is_placeholder,
            model_version=self.name,
        )

    # --- fitted-parameter setters ------------------------------------

    def set_weights(
        self,
        weights: Sequence[float],
        *,
        decision_threshold: float | None = None,
    ) -> "CombinerDetector":
        """Switch to ``method="weighted"`` with a fitted weight vector (and,
        optionally, its tuned threshold). Returns ``self``."""
        if len(weights) != len(self._members):
            raise ValueError("weights must match the number of members")
        self._weights = [float(w) for w in weights]
        self.method = "weighted"
        self.name = f"{self.name_prefix}-weighted"
        if decision_threshold is not None:
            self.decision_threshold = float(decision_threshold)
        return self

    def attach_meta_classifier(
        self,
        meta_classifier: Any,
        *,
        decision_threshold: float | None = None,
    ) -> "CombinerDetector":
        """Swap in a trained meta-classifier and switch ``method`` to ``"meta"``."""
        self._meta = meta_classifier
        self.method = "meta"
        self.name = f"{self.name_prefix}-meta"
        if decision_threshold is not None:
            self.decision_threshold = float(decision_threshold)
        return self
