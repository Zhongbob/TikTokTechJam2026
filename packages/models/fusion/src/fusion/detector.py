"""Fusion detector — combine several single-model detectors into one verdict.

The two members it is built for:

* **Community-Forensics** (`community_forensics.CommunityForensicsDetector`) —
  a whole-image ViT that catches *fully synthetic* (text-to-image) images.
* **PSCC-Net** (`pscc_net.PSCCNetDetector`) — catches *locally tampered* images
  (splice / copy-move / inpainting) that a whole-image classifier misses.

They have complementary blind spots, so the default decider is deliberately
dumb: **take the max of the members' p(ai) and compare it to a threshold** — if
*either* model is confident the image is fake, the fusion says fake.

    detector = FusionDetector.use_default()            # CF + PSCC-Net, combine="max"
    result   = detector.predict(pil_image)             # DetectionResult w/ per-member breakdown
    metrics  = detector.evaluate(val_samples, generate_confusion_matrix=True)

`combine`:
    * ``"max"``   (default) — ``max(member p(ai))``
    * ``"mean"``  — unweighted average
    * ``"weighted"`` — needs ``weights=[...]`` (same length/order as members)
    * ``"meta"``  — feed the member score vector to a trained meta-classifier
      (see `fusion.trainer.FusionTrainer`); raises until one is attached.

`FusionDetector` implements `detector_common.ImageDetector`, so `predict()` /
`evaluate()` match every other detector in this repo. It does **not** load any
weights of its own — it just orchestrates its members.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from detector_common import ImageDetector
from PIL import Image
from shared_types.detection import DetectionResult, EnsembleMemberResult

_COMBINE_MODES = {"max", "mean", "weighted", "meta"}


class FusionDetector(ImageDetector):
    """Combines member detectors' p(ai) into one score + verdict."""

    name = "fusion-max"
    is_placeholder = False

    def __init__(
        self,
        members: Sequence[Any],
        *,
        combine: str = "max",
        weights: Sequence[float] | None = None,
        decision_threshold: float = 0.5,
        meta_classifier: Any | None = None,
        name: str | None = None,
    ) -> None:
        if not members:
            raise ValueError("FusionDetector needs at least one member detector")
        if combine not in _COMBINE_MODES:
            raise ValueError(f"combine must be one of {sorted(_COMBINE_MODES)}")
        if combine == "weighted":
            if weights is None or len(weights) != len(members):
                raise ValueError("combine='weighted' needs weights= matching members")
        if combine == "meta" and meta_classifier is None:
            raise ValueError(
                "combine='meta' needs a trained meta_classifier — train one with "
                "fusion.trainer.FusionTrainer (or use combine='max')."
            )

        self._members = list(members)
        self.combine = combine
        self._weights = [float(w) for w in weights] if weights is not None else None
        self.decision_threshold = decision_threshold
        self._meta = meta_classifier
        self.name = name or f"fusion-{combine}"

    # --- construction --------------------------------------------------

    @classmethod
    def from_members(cls, members: Sequence[Any], **kwargs: Any) -> "FusionDetector":
        """Wrap an explicit list of already-constructed member detectors."""
        return cls(members, **kwargs)

    @classmethod
    def use_default(
        cls,
        *,
        device: str = "auto",
        combine: str = "max",
        decision_threshold: float = 0.5,
        **member_kwargs: Any,
    ) -> "FusionDetector":
        """Build the default fusion: Community-Forensics + PSCC-Net.

        Downloads each member's weights on first use (see their own
        ``use_default``). ``member_kwargs`` is forwarded to *both* members.
        """
        from community_forensics import CommunityForensicsDetector
        from pscc_net import PSCCNetDetector

        members = [
            CommunityForensicsDetector.use_default(device=device),
            PSCCNetDetector.use_default(device=device, **member_kwargs),
        ]
        return cls(members, combine=combine, decision_threshold=decision_threshold)

    # --- scoring -----------------------------------------------------

    def member_predictions(self, image: Image.Image) -> list[DetectionResult]:
        """Run every member on one image and return their raw `DetectionResult`s
        (order matches ``self.members``). Useful for meta-classifier feature
        extraction and for debugging which member fired."""
        rgb = image.convert("RGB")
        return [member.predict(rgb) for member in self._members]

    @property
    def members(self) -> list[Any]:
        return list(self._members)

    def _combine_scores(self, probs: Sequence[float]) -> float:
        if self.combine == "max":
            return max(probs)
        if self.combine == "mean":
            return sum(probs) / len(probs)
        if self.combine == "weighted":
            assert self._weights is not None
            total = sum(self._weights) or 1.0
            return sum(p * w for p, w in zip(probs, self._weights)) / total
        # "meta"
        return float(self._meta.predict_proba([list(probs)])[0][1])

    def _score(self, image: Image.Image) -> float:
        probs = [r.ai_generated_probability for r in self.member_predictions(image)]
        return min(1.0, max(0.0, self._combine_scores(probs)))

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

        p_ai = min(1.0, max(0.0, self._combine_scores(probs)))
        return DetectionResult(
            verdict="ai_generated" if p_ai >= self.decision_threshold else "real",
            ai_generated_probability=p_ai,
            member_results=member_results,
            is_placeholder=self.is_placeholder,
            model_version=self.name,
        )

    # --- meta-classifier hook (future) --------------------------------

    def attach_meta_classifier(self, meta_classifier: Any) -> None:
        """Swap in a trained meta-classifier and switch ``combine`` to ``"meta"``.
        Called by `fusion.trainer.FusionTrainer` once that is implemented."""
        self._meta = meta_classifier
        self.combine = "meta"
        self.name = "fusion-meta"
