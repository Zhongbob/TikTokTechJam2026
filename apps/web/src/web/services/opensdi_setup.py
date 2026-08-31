"""One-time OpenSDI provisioning, run when the Streamlit server boots.

The ensemble's ``fusion`` member wraps OpenSDI / MaskCLIP, which can't run from a
bare install — it needs the ``iamwangyabin/OpenSDI`` repo cloned plus the
MaskCLIP / MAE checkpoints downloaded. ``opensdi_detector.setup_opensdi()`` does
all of that; we call it once here (``@st.cache_resource`` => once per process)
before the app renders, and export ``$OPENSDI_REPO`` so
``services.factory.get_detector`` picks the clone up.

Setup is best-effort: if it fails (offline, ``pip`` unavailable in the venv,
IMDLBenCo's ``numpy<2`` pin, ...) the app keeps working — ``factory.py`` falls
back to the ensemble without its fusion member, then to the placeholder.

Set ``WEB_SKIP_OPENSDI_SETUP=1`` to skip this entirely (e.g. when the host has
already provisioned OpenSDI and exported ``$OPENSDI_REPO``).
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

#: apps/web/src/web/services/opensdi_setup.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[5]


@st.cache_resource(show_spinner="Setting up the OpenSDI fusion member (first launch only)…")
def ensure_opensdi_ready() -> str | None:
    """Run ``setup_opensdi()`` once. Returns the OpenSDI repo dir, or ``None``."""
    if os.environ.get("WEB_SKIP_OPENSDI_SETUP"):
        return os.environ.get("OPENSDI_REPO")

    try:
        from opensdi_detector import setup_opensdi
    except Exception as error:  # noqa: BLE001 - package missing entirely
        st.warning(f"opensdi_detector unavailable ({error}); skipping OpenSDI setup.")
        return None

    try:
        info = setup_opensdi(repo_root=str(_REPO_ROOT))
    except Exception as error:  # noqa: BLE001 - clone / pip / download failure
        st.warning(
            f"OpenSDI setup didn't complete ({error}). The ensemble will run "
            "without its fusion member. To fix: provision OpenSDI in an env with "
            "`pip` and `numpy<2`, then set $OPENSDI_REPO."
        )
        return None

    repo_dir = info["repo_dir"]
    os.environ["OPENSDI_REPO"] = repo_dir
    return repo_dir
