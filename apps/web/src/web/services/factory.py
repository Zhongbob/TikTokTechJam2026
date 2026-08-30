"""Single point of construction for the pipeline's model stages.

THE SWAP SEAM: when a real model for a stage is ready, replace the Dummy*
instantiation below with an import from packages/models/*. Nothing else in the
app needs to change, since every implementation satisfies the
shared_types.interfaces Protocols (AutoencoderRestorer / EnsembleDetector).

The "Normal Classifier" method is already swapped in — get_detector() returns
the trained NormalClassifierDetector for it.
"""

from __future__ import annotations

import streamlit as st
from shared_types.interfaces import AutoencoderRestorer, EnsembleDetector

from web.services.methods import NORMAL_CLASSIFIER
from web.services.placeholder_models import DummyAutoencoderRestorer, DummyEnsembleDetector


@st.cache_resource
def get_restorer() -> AutoencoderRestorer:
    return DummyAutoencoderRestorer()
    # SWAP POINT: from autoencoder import RealAutoencoderRestorer
    #             return RealAutoencoderRestorer()


@st.cache_resource
def get_detector(method: str) -> EnsembleDetector:
    """The detector for the chosen method.

    - ``normal_classifier`` -> the trained ``NormalClassifierDetector`` loaded
      from its bundled default checkpoint.
    - ``transform_reversal`` -> the ensemble detector (still a placeholder).
    """
    if method == NORMAL_CLASSIFIER:
        try:
            from normal_classifier import NormalClassifierDetector
        except ImportError as error:  # pragma: no cover - env-dependent
            raise RuntimeError(
                "The 'normal_classifier' package isn't installed. Run `uv sync` in apps/web."
            ) from error
        return NormalClassifierDetector.use_default()

    return DummyEnsembleDetector()
    # SWAP POINT (transform_reversal): from ensemble import RealEnsembleDetector
    #                                  return RealEnsembleDetector()
