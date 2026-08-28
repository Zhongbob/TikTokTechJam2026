import streamlit as st

from web.app import render_app

st.set_page_config(
    page_title="AI-Generated Image Detection",
    page_icon="🕵️",
    layout="wide",
)

render_app()
