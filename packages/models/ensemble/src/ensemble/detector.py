"""Ensemble detector — the fusion model + the standalone trained classifiers,
combined into one verdict.

Members (default, in order):

1. **fusion** (`fusion.FusionDetector`) — CF + OpenSDI (a `CombinerDetector`
   itself). Optionally fed an autoencoder-*restored* image (see
   ``use_autoencoder``); every other member still sees the original image.
2. **convnext_aigc** (`convnext_aigc.ConvNextAIGCDetector`)
3. **clip_vit_b32** (`clip_vit_b32.ClipViTB32Detector`)
4. **dinov2** (`dinov2.DINOv2Detector`)
5. **normal_classifier** (`normal_classifier.NormalClassifierDetector`)
6. **swin** (`swin.SwinDetector`)

The combination logic (``method`` ∈ max / mean / weighted / meta, per-member
results, ``predict`` / ``evaluate``) is inherited from
`detector_common.CombinerDetector`; fit ``weighted`` / ``meta`` with
`ensemble.trainer.EnsembleTrainer`.

    det = EnsembleDetector.use_default(opensdi_repo_dir="/content/OpenSDI",
                                       use_autoencoder=True)
    det.predict(pil_image)
    det.evaluate(val_samples, generate_confusion_matrix=True)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from detector_common import CombinerDetector
from PIL import Image

DEFAULT_MEMBERS = (
    "fusion", "convnext_aigc", "clip_vit_b32", "dinov2", "normal_classifier", "swin",
)


class _RestoreBeforePredict:
    """Adapter: run ``restorer.predict(image)`` (autoencoder) before delegating
    to ``inner.predict``. Looks like a member detector to `CombinerDetector`."""

    def __init__(self, inner: Any, restorer: Any) -> None:
        self._inner = inner
        self._restorer = restorer
        self.name = f"{getattr(inner, 'name', 'member')}+autoencoder"
        self.is_placeholder = bool(getattr(inner, "is_placeholder", False))

    def predict(self, image: Image.Image) -> Any:
        return self._inner.predict(self._restorer.predict(image))


def _build_restorer(checkpoint: str | Path | None, device: str) -> Any:
    from autoencoder import AutoencoderRestorer

    if checkpoint is not None:
        return AutoencoderRestorer.from_checkpoint(str(checkpoint), device=device)
    return AutoencoderRestorer.use_default()


def _build_member(name: str, *, device: str, **kwargs: Any) -> Any:
    """``kwargs`` (from ``member_kwargs[name]``) are forwarded to the member's
    ``use_default`` — e.g. ``checkpoint=`` / ``positive_class=`` / ``flip=``."""
    if name == "fusion":
        raise ValueError("build the 'fusion' member via build_default_ensemble_members")
    if name == "convnext_aigc":
        from convnext_aigc import ConvNextAIGCDetector

        return ConvNextAIGCDetector.use_default(device=device, **kwargs)
    if name == "clip_vit_b32":
        from clip_vit_b32 import ClipViTB32Detector

        return ClipViTB32Detector.use_default(device=device, **kwargs)
    if name == "dinov2":
        from dinov2 import DINOv2Detector

        return DINOv2Detector.use_default(device=device, **kwargs)
    if name == "normal_classifier":
        from normal_classifier import NormalClassifierDetector

        return NormalClassifierDetector.use_default(**kwargs)
    if name == "swin":
        from swin import SwinDetector

        return SwinDetector.use_default(device=device, **kwargs)
    raise ValueError(f"unknown ensemble member {name!r}")


def build_default_ensemble_members(
    *,
    device: str = "auto",
    include: Sequence[str] = DEFAULT_MEMBERS,
    use_autoencoder: bool = False,
    autoencoder_checkpoint: str | Path | None = None,
    autoencoder_device: str = "cpu",
    fusion_kwargs: dict[str, Any] | None = None,
    opensdi_repo_dir: str | Path | None = None,
    member_kwargs: dict[str, dict[str, Any]] | None = None,
) -> list[Any]:
    """Construct the ensemble's member detectors (shared by
    `EnsembleDetector.use_default` and `EnsembleTrainer.use_default`).

    ``use_autoencoder`` wraps **only the fusion member** so it receives a
    restored image; the other members get the original. ``fusion_kwargs`` is
    forwarded to `fusion.FusionDetector.use_default` (e.g. ``method``,
    ``weights``, ``opensdi_*``); ``opensdi_repo_dir`` is a convenience shortcut.

    ``member_kwargs`` = ``{member_name: {...}}`` forwarded to each member's
    ``use_default`` — e.g. ``{"dinov2": {"checkpoint": "/content/dino.pt"},
    "swin": {"checkpoint": "/content/swin.pth"}}``. (For ``"fusion"`` put the
    args in ``fusion_kwargs`` instead.)
    """
    restorer = _build_restorer(autoencoder_checkpoint, autoencoder_device) if use_autoencoder else None
    mkw = member_kwargs or {}

    members: list[Any] = []
    for name in include:
        if name == "fusion":
            from fusion import FusionDetector

            fk = dict(fusion_kwargs or {})
            if opensdi_repo_dir is not None:
                fk.setdefault("opensdi_repo_dir", opensdi_repo_dir)
            fusion = FusionDetector.use_default(device=device, **fk)
            members.append(_RestoreBeforePredict(fusion, restorer) if restorer is not None else fusion)
        else:
            members.append(_build_member(name, device=device, **mkw.get(name, {})))
    return members


class EnsembleDetector(CombinerDetector):
    """Fusion + convnext + clip + dinov2 + normal_classifier + swin, combined."""

    name = "ensemble-max"
    name_prefix = "ensemble"
    default_max_threshold = 0.5  # no SID-tuned value yet — fit weighted/meta

    @classmethod
    def use_default(
        cls,
        *,
        device: str = "auto",
        method: str = "max",
        weights: list[float] | None = [0.0833, 0.0, 0.1667, 0.1667, 0.4167, 0.1667],
        decision_threshold: float | None = None,
        include: Sequence[str] = DEFAULT_MEMBERS,
        use_autoencoder: bool = False,
        autoencoder_checkpoint: str | Path | None = None,
        autoencoder_device: str = "cpu",
        fusion_kwargs: dict[str, Any] | None = None,
        opensdi_repo_dir: str | Path | None = None,
        member_kwargs: dict[str, dict[str, Any]] | None = None,
    ) -> "EnsembleDetector":
        """Build the full ensemble. Each member downloads / loads its own weights
        on first use (see their ``use_default``); OpenSDI (inside the fusion
        member) needs ``opensdi_detector.setup_opensdi()`` first.

        ``use_autoencoder=True`` runs the autoencoder over the image before the
        **fusion** member only. ``member_kwargs={"dinov2": {"checkpoint": ...}}``
        overrides a member's checkpoint / options.
        """
        members = build_default_ensemble_members(
            device=device,
            include=include,
            use_autoencoder=use_autoencoder,
            autoencoder_checkpoint=autoencoder_checkpoint,
            autoencoder_device=autoencoder_device,
            fusion_kwargs=fusion_kwargs,
            opensdi_repo_dir=opensdi_repo_dir,
            member_kwargs=member_kwargs,
        )
        return cls(members, method=method, weights=weights, decision_threshold=decision_threshold)
