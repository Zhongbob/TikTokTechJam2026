import streamlit as st

from web.app import render_app
from web.services.opensdi_setup import ensure_opensdi_ready

st.set_page_config(
    page_title="AI-Generated Image Detection",
    page_icon="🕵️",
    layout="wide",
)

# Provision the OpenSDI fusion member before the app renders (once per process).
ensure_opensdi_ready()

render_app()
