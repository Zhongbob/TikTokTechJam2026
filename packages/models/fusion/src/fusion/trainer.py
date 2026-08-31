"""Trainer for `FusionDetector`'s combination parameters.

Thin wrapper over `detector_common.CombinerTrainer` (which holds all the
weight-grid / threshold / meta-classifier / `compare_methods` logic) — this
module just builds the default CF + OpenSDI members.

    trainer  = FusionTrainer.use_default(opensdi_repo_dir="/content/OpenSDI")
    table    = trainer.compare_methods(train_samples, val_samples)
    detector = trainer.as_detector(method="meta")     # or "weighted" / "max"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from detector_common import CombinerTrainer

from fusion.detector import FusionDetector, build_default_members

# re-exported for callers that still import the helpers from here
from detector_common.combiner_trainer import (  # noqa: F401
    _classify_report,
    _fuse,
    _metrics_at,
    _pick_threshold,
    _weight_grid,
)


class FusionTrainer(CombinerTrainer):
    """Fits `FusionDetector`'s weights / meta-classifier on labelled data."""

    name = "fusion-weight-trainer"
    detector_cls = FusionDetector
    _label = "fusion"

    @classmethod
    def use_default(
        cls,
        *,
        device: str = "auto",
        opensdi_repo_dir: str | Path | None = None,
        opensdi_weights_dir: str | Path | None = None,
        opensdi_checkpoint: str | Path | None = None,
        opensdi_kwargs: dict[str, Any] | None = None,
    ) -> "FusionTrainer":
        """Build with the same two members as `FusionDetector.use_default`."""
        members = build_default_members(
            device=device,
            opensdi_repo_dir=opensdi_repo_dir,
            opensdi_weights_dir=opensdi_weights_dir,
            opensdi_checkpoint=opensdi_checkpoint,
            opensdi_kwargs=opensdi_kwargs,
        )
        return cls(members)
