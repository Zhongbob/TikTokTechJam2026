"""Fusion detector — Community-Forensics + OpenSDI combined into one verdict.

The combination logic (methods max / mean / weighted / meta, per-member results,
`predict` / `evaluate`) lives in `detector_common.CombinerDetector`; this module
just wires the two default members and the SID-tuned operating point.

* **Community-Forensics** (`community_forensics.CommunityForensicsDetector`) — a
  whole-image ViT that catches *fully synthetic* images; blind to local edits.
* **OpenSDI / MaskCLIP** (`opensdi_detector.OpenSDIDetector`) — a
  diffusion-inpainting *localizer* that catches *locally tampered* images; blind
  to fully-synthetic ones.

    detector = FusionDetector.use_default(opensdi_repo_dir="/path/to/OpenSDI")
    result   = detector.predict(pil_image)
    metrics  = detector.evaluate(val_samples, generate_confusion_matrix=True)

``method="max"`` (default) uses ``DEFAULT_MAX_THRESHOLD`` (0.19, tuned on SID);
fit ``weighted`` / ``meta`` with `fusion.trainer.FusionTrainer`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from detector_common import CombinerDetector

#: Operating threshold tuned for ``method="max"`` on SID (CF + OpenSDI).
DEFAULT_MAX_THRESHOLD = 0.19

#: Module-level default OpenSDI clone for `use_default` / `FusionTrainer.use_default`.
#: Set once (``fusion.detector.DEFAULT_OPENSDI_REPO_DIR = "/content/OpenSDI"``)
#: instead of passing ``opensdi_repo_dir=`` each call. ``None`` -> fall back to
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
    """Construct the default fusion members — Community-Forensics + OpenSDI.
    Shared by `FusionDetector.use_default` and `FusionTrainer.use_default`.

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


class FusionDetector(CombinerDetector):
    """Community-Forensics + OpenSDI, combined via `CombinerDetector`."""

    name = "fusion-max"
    name_prefix = "fusion"
    default_max_threshold = DEFAULT_MAX_THRESHOLD

    @classmethod
    def use_default(
        cls,
        *,
        device: str = "auto",
        method: str = "max",
        weights: list[float] | None = None,
        decision_threshold: float | None = None,
        opensdi_repo_dir: str | Path | None = None,
        opensdi_weights_dir: str | Path | None = None,
        opensdi_checkpoint: str | Path | None = None,
        opensdi_kwargs: dict[str, Any] | None = None,
    ) -> "FusionDetector":
        """Build the default fusion (CF + OpenSDI).

        OpenSDI needs a one-time ``opensdi_detector.setup_opensdi()``. Point at
        the clone via ``opensdi_repo_dir=`` here / ``DEFAULT_OPENSDI_REPO_DIR`` /
        ``$OPENSDI_REPO`` / ``/content/OpenSDI``.

        ``decision_threshold=None`` -> ``DEFAULT_MAX_THRESHOLD`` (0.19) for
        ``method="max"``, else 0.5. For ``method="weighted"`` pass ``weights=``.
        """
        members = build_default_members(
            device=device,
            opensdi_repo_dir=opensdi_repo_dir,
            opensdi_weights_dir=opensdi_weights_dir,
            opensdi_checkpoint=opensdi_checkpoint,
            opensdi_kwargs=opensdi_kwargs,
        )
        return cls(members, method=method, weights=weights, decision_threshold=decision_threshold)
