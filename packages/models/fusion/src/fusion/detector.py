"""Fusion detector — combine several single-model detectors into one verdict.

The two members it is built for:

* **Community-Forensics** (`community_forensics.CommunityForensicsDetector`) —
  a whole-image ViT that catches *fully synthetic* (text-to-image) images.
  Near-perfect on SID's `synthetic` class, blind to local edits.
* **OpenSDI / MaskCLIP** (`opensdi_detector.OpenSDIDetector`) — a
  diffusion-inpainting *localizer* that catches *locally tampered* images
  (SID's `tampered` class: real photos with an SD-inpainted region), which a
  whole-image classifier misses. Blind to fully-synthetic images.

They have complementary blind spots. `method` selects how the member scores are
fused into one `p(ai)`, which is then compared to `decision_threshold`:

    * ``"max"`` / ``"threshold"`` (default) — ``max(member p(ai))``. The simple
      threshold split: fake if *either* member exceeds the threshold. Works
      because each member scores ~0 outside its own domain. Tuned threshold on
      SID (CF + OpenSDI) ≈ 0.19 (`DEFAULT_MAX_THRESHOLD`).
    * ``"mean"`` — unweighted average.
    * ``"weighted"`` — ``sum(w_i * p_i) / sum(w_i)``; needs ``weights=[...]``
      matching the members. Fit them with `fusion.trainer.FusionTrainer`.
    * ``"meta"`` — feed the member score vector to a trained meta-classifier
      (still a stub in the trainer); raises until one is attached.

    detector = FusionDetector.use_default(opensdi_repo_dir="/path/to/OpenSDI")
    result   = detector.predict(pil_image)             # DetectionResult w/ per-member breakdown
    metrics  = detector.evaluate(val_samples, generate_confusion_matrix=True)

`FusionDetector` implements `detector_common.ImageDetector`, so `predict()` /
`evaluate()` match every other detector in this repo. It does **not** load any
weights of its own — it just orchestrates its members.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from detector_common import ImageDetector
from PIL import Image
from shared_types.detection import DetectionResult, EnsembleMemberResult

#: Public fusion methods. ``"threshold"`` is an alias for ``"max"``.
_METHODS = {"max", "mean", "weighted", "meta"}
_METHOD_ALIASES = {"threshold": "max"}

#: Operating threshold tuned for method="max" on SID (CF + OpenSDI). `use_default`
#: falls back to this for the max method; override per call.
DEFAULT_MAX_THRESHOLD = 0.19

#: Module-level default for the OpenSDI clone used by ``FusionDetector.use_default``
#: and ``FusionTrainer.use_default``. Set it once
#: (``fusion.detector.DEFAULT_OPENSDI_REPO_DIR = "/content/OpenSDI"``) instead of
#: passing ``opensdi_repo_dir=`` every call. ``None`` -> fall back to
#: ``$OPENSDI_REPO`` / ``/content/OpenSDI`` (handled by OpenSDIDetector).
DEFAULT_OPENSDI_REPO_DIR: str | Path | None = None


def build_default_members(
    *,
    device: str = "auto",
    opensdi_repo_dir: str | Path | None = None,
    opensdi_weights_dir: str | Path | None = None,
    opensdi_checkpoint: str | Path | None = None,
    opensdi_kwargs: dict[str, Any] | None = None,
) -> list[Any]:
    """Construct the default fusion members — Community-Forensics + OpenSDI —
    exactly as `FusionDetector.use_default` does. Shared with
    `fusion.trainer.FusionTrainer.use_default` so both build them identically.

    OpenSDI is wired as a tamper *localizer* (``score_mode="mask"`` /
    ``mask_reduce="max"``). Repo resolution priority: ``opensdi_repo_dir`` arg,
    then ``DEFAULT_OPENSDI_REPO_DIR``, then ``$OPENSDI_REPO`` / ``/content/OpenSDI``.
    """
    from community_forensics import CommunityForensicsDetector
    from opensdi_detector import OpenSDIDetector

    opensdi_args: dict[str, Any] = {"score_mode": "mask", "mask_reduce": "max"}
    repo = opensdi_repo_dir or DEFAULT_OPENSDI_REPO_DIR
    if repo is not None:
        opensdi_args["repo_dir"] = repo
    if opensdi_weights_dir is not None:
        opensdi_args["weights_dir"] = opensdi_weights_dir
    if opensdi_checkpoint is not None:
        opensdi_args["checkpoint"] = opensdi_checkpoint
    opensdi_args.update(opensdi_kwargs or {})

    return [
        CommunityForensicsDetector.use_default(device=device),
        OpenSDIDetector.use_default(device=device, **opensdi_args),
    ]


class FusionDetector(ImageDetector):
    """Combines member detectors' p(ai) into one score + verdict."""

    name = "fusion-max"
    is_placeholder = False

    def __init__(
        self,
        members: Sequence[Any],
        *,
        method: str = "max",
        weights: Sequence[float] | None = None,
        decision_threshold: float = 0.5,
        meta_classifier: Any | None = None,
        name: str | None = None,
    ) -> None:
        if not members:
            raise ValueError("FusionDetector needs at least one member detector")

        method = _METHOD_ALIASES.get(method, method)
        if method not in _METHODS:
            raise ValueError(
                f"method must be one of {sorted(_METHODS)} (or 'threshold' == 'max')"
            )
        if method == "weighted":
            if weights is None or len(weights) != len(members):
                raise ValueError("method='weighted' needs weights= matching members")
        if method == "meta" and meta_classifier is None:
            raise ValueError(
                "method='meta' needs a trained meta_classifier — train one with "
                "fusion.trainer.FusionTrainer (or use method='max' / 'weighted')."
            )

        self._members = list(members)
        self.method = method
        self._weights = [float(w) for w in weights] if weights is not None else None
        self.decision_threshold = decision_threshold
        self._meta = meta_classifier
        self.name = name or f"fusion-{method}"

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
        method: str = "max",
        weights: Sequence[float] | None = None,
        decision_threshold: float | None = None,
        opensdi_repo_dir: str | Path | None = None,
        opensdi_weights_dir: str | Path | None = None,
        opensdi_checkpoint: str | Path | None = None,
        opensdi_kwargs: dict[str, Any] | None = None,
    ) -> "FusionDetector":
        """Build the default fusion: Community-Forensics + OpenSDI.

        Community-Forensics downloads its weights on first use. OpenSDI needs a
        one-time ``opensdi_detector.setup_opensdi()`` (clones the repo, installs
        ``IMDLBenCo`` + OpenAI ``clip``, downloads the ~3.1 GB MaskCLIP
        checkpoint).

        Point at the OpenSDI clone with, in priority order: ``opensdi_repo_dir=``
        here, ``fusion.detector.DEFAULT_OPENSDI_REPO_DIR``, ``$OPENSDI_REPO``, or
        ``/content/OpenSDI``. ``opensdi_weights_dir`` / ``opensdi_checkpoint``
        override where the ``.pth`` is found.

        ``decision_threshold=None`` picks a sensible default per method
        (``DEFAULT_MAX_THRESHOLD`` for ``"max"``, else 0.5). For
        ``method="weighted"`` pass ``weights=`` (from `FusionTrainer`).
        """
        members = build_default_members(
            device=device,
            opensdi_repo_dir=opensdi_repo_dir,
            opensdi_weights_dir=opensdi_weights_dir,
            opensdi_checkpoint=opensdi_checkpoint,
            opensdi_kwargs=opensdi_kwargs,
        )
        canonical = _METHOD_ALIASES.get(method, method)
        if decision_threshold is None:
            decision_threshold = DEFAULT_MAX_THRESHOLD if canonical == "max" else 0.5
        return cls(
            members,
            method=method,
            weights=weights,
            decision_threshold=decision_threshold,
        )

    # --- scoring -----------------------------------------------------

    def member_predictions(self, image: Image.Image) -> list[DetectionResult]:
        """Run every member on one image and return their raw `DetectionResult`s
        (order matches ``self.members``). Useful for weight/meta fitting and for
        debugging which member fired."""
        rgb = image.convert("RGB")
        return [member.predict(rgb) for member in self._members]

    @property
    def members(self) -> list[Any]:
        return list(self._members)

    @property
    def weights(self) -> list[float] | None:
        return list(self._weights) if self._weights is not None else None

    def fuse_scores(self, probs: Sequence[float]) -> float:
        """Combine a member p(ai) vector into one score, per ``self.method``."""
        if self.method == "max":
            return max(probs)
        if self.method == "mean":
            return sum(probs) / len(probs)
        if self.method == "weighted":
            assert self._weights is not None
            total = sum(self._weights) or 1.0
            return sum(p * w for p, w in zip(probs, self._weights)) / total
        # "meta"
        return float(self._meta.predict_proba([list(probs)])[0][1])

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
    ) -> "FusionDetector":
        """Switch to ``method="weighted"`` with a fitted weight vector (and,
        optionally, its tuned threshold). Returns ``self``."""
        if len(weights) != len(self._members):
            raise ValueError("weights must match the number of members")
        self._weights = [float(w) for w in weights]
        self.method = "weighted"
        self.name = "fusion-weighted"
        if decision_threshold is not None:
            self.decision_threshold = float(decision_threshold)
        return self

    def attach_meta_classifier(self, meta_classifier: Any) -> None:
        """Swap in a trained meta-classifier and switch ``method`` to ``"meta"``."""
        self._meta = meta_classifier
        self.method = "meta"
        self.name = "fusion-meta"
