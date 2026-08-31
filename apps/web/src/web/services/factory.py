"""Single point of construction for the pipeline's model stages.

Both methods now use real models; the ``Dummy*`` classes in
``placeholder_models`` are only a graceful fallback when a package / checkpoint
isn't available in the current environment.

- **Normal Classifier** -> the trained ``YoloDetector``.
- **Transform Reversal** -> the ``EnsembleDetector`` (``use_autoencoder=True``:
  its fusion sub-model scores an autoencoder-restored image, the other members
  see the original), plus the real ``AutoencoderRestorer`` for the preview panel.
"""

from __future__ import annotations

import os

import streamlit as st
from shared_types.interfaces import AutoencoderRestorer, EnsembleDetector

from web.services.methods import YOLO_CLASSIFIER
from web.services.placeholder_models import DummyAutoencoderRestorer, DummyEnsembleDetector


@st.cache_resource
def get_restorer() -> AutoencoderRestorer:
    """The autoencoder restorer used for the preview panel."""
    try:
        from autoencoder import AutoencoderRestorer as RealRestorer

        return RealRestorer.use_default()
    except Exception as error:  # noqa: BLE001 - missing package / checkpoint
        st.warning(f"Autoencoder restorer unavailable ({error}); using a placeholder filter.")
        return DummyAutoencoderRestorer()


@st.cache_resource
def get_detector(method: str) -> EnsembleDetector:
    """The detector for the chosen method."""
    if method == YOLO_CLASSIFIER:
        try:
            from yolo import YoloDetector
        except ImportError as error:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "The 'yolo' package isn't installed. Run `uv sync` in apps/web."
            ) from error
        return YoloDetector.use_default()

    # transform_reversal -> the ensemble, autoencoder ON (fusion member only).
    try:
        from ensemble import DEFAULT_MEMBERS, EnsembleDetector as RealEnsemble
    except ImportError as error:
        st.warning(f"'ensemble' package not installed ({error}); using a placeholder detector.")
        return DummyEnsembleDetector()

    opensdi_repo = os.environ.get("OPENSDI_REPO")
    try:
        return RealEnsemble.use_default(use_autoencoder=True, opensdi_repo_dir=opensdi_repo)
    except Exception as error:  # noqa: BLE001 - usually the fusion member (OpenSDI) not set up
        st.warning(
            f"Full ensemble unavailable ({error}) — running without the OpenSDI fusion member. "
            "Run `python -m opensdi_detector.bootstrap` and set $OPENSDI_REPO for the full model."
        )
        try:
            members = [m for m in DEFAULT_MEMBERS if m != "fusion"]
            return RealEnsemble.use_default(use_autoencoder=False, include=members)
        except Exception as inner:  # noqa: BLE001
            st.warning(f"Ensemble members unavailable ({inner}); using a placeholder detector.")
            return DummyEnsembleDetector()
