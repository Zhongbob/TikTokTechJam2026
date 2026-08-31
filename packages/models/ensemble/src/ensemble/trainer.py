"""Trainer for `EnsembleDetector`'s combination parameters.

Thin wrapper over `detector_common.CombinerTrainer` — all the weight-grid /
threshold / tree-based meta-classifier / `compare_methods` logic is shared with
`fusion.FusionTrainer`.

    tr       = EnsembleTrainer.use_default(opensdi_repo_dir="/content/OpenSDI",
                                           use_autoencoder=True)
    Xtr, ytr = tr.member_score_matrix(train_samples)   # augmented SID
    Xva, yva = tr.member_score_matrix(val_samples)
    tr.compare_methods(X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva,
                       meta_kinds=("tree", "gboost"))

    tr.fit_meta_classifier(X=Xtr, y=ytr, kind="gboost")   # or optimal_weights(...)
    det = tr.as_detector(method="meta")                   # -> EnsembleDetector
    tr.save("ensemble_meta.json")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from detector_common import CombinerTrainer

from ensemble.detector import DEFAULT_MEMBERS, EnsembleDetector, build_default_ensemble_members


class EnsembleTrainer(CombinerTrainer):
    """Fits `EnsembleDetector`'s weights / meta-classifier on labelled data."""

    name = "ensemble-weight-trainer"
    detector_cls = EnsembleDetector
    _label = "ensemble"

    @classmethod
    def use_default(
        cls,
        *,
        device: str = "auto",
        include: Sequence[str] = DEFAULT_MEMBERS,
        use_autoencoder: bool = False,
        autoencoder_checkpoint: str | Path | None = None,
        autoencoder_device: str = "cpu",
        fusion_kwargs: dict[str, Any] | None = None,
        opensdi_repo_dir: str | Path | None = None,
    ) -> "EnsembleTrainer":
        """Build with the same members as `EnsembleDetector.use_default` — pass
        the SAME ``use_autoencoder`` / member args you'll use at inference so the
        fitted weights match."""
        members = build_default_ensemble_members(
            device=device,
            include=include,
            use_autoencoder=use_autoencoder,
            autoencoder_checkpoint=autoencoder_checkpoint,
            autoencoder_device=autoencoder_device,
            fusion_kwargs=fusion_kwargs,
            opensdi_repo_dir=opensdi_repo_dir,
        )
        return cls(members)
