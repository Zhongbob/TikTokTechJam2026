"""Typed accessors around st.session_state for this app's pipeline state."""

from __future__ import annotations

import streamlit as st
from web.services.pipeline import PipelineResult

_RESULT_KEY = "pipeline_result"


def init_session_state() -> None:
    if _RESULT_KEY not in st.session_state:
        st.session_state[_RESULT_KEY] = None


def get_pipeline_result() -> PipelineResult | None:
    return st.session_state.get(_RESULT_KEY)


def set_pipeline_result(result: PipelineResult) -> None:
    st.session_state[_RESULT_KEY] = result
