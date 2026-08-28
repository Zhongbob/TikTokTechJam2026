"""Single point of construction for the pipeline's model stages.

THE SWAP SEAM: when the real autoencoder/ensemble packages are ready,
replace the Dummy* instantiations below with imports from
packages/models/autoencoder and packages/models/ensemble. Nothing else in
the app needs to change, since both satisfy the shared_types.interfaces
Protocols (AutoencoderRestorer / EnsembleDetector).
"""

from __future__ import annotations

import streamlit as st
from shared_types.interfaces import AutoencoderRestorer, EnsembleDetector

from web.services.placeholder_models import DummyAutoencoderRestorer, DummyEnsembleDetector


@st.cache_resource
def get_restorer() -> AutoencoderRestorer:
    return DummyAutoencoderRestorer()
    # SWAP POINT: from autoencoder import RealAutoencoderRestorer
    #             return RealAutoencoderRestorer()


@st.cache_resource
def get_detector() -> EnsembleDetector:
    return DummyEnsembleDetector()
    # SWAP POINT: from ensemble import RealEnsembleDetector
    #             return RealEnsembleDetector()
