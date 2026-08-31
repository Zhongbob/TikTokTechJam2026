from __future__ import annotations

import streamlit as st

from web import state
from web.components import detection_chart, preview, results, sidebar
from web.services.methods import METHOD_LABELS
from web.services.pipeline import run_pipeline
from web.services.transforms import apply_transform_pipeline


def _run_and_store(image, pipeline, source_label: str, method: str) -> None:
    """Run one pipeline pass and stash the result, surfacing model-loading
    problems (missing checkpoint, package not installed) as a friendly error
    rather than a raw traceback."""
    try:
        result = run_pipeline(image, pipeline, source_label, method)
    except (FileNotFoundError, RuntimeError) as error:
        st.error(f"Couldn't run **{METHOD_LABELS[method]}**: {error}")
        return
    state.set_pipeline_result(result)


def render_app() -> None:
    state.init_session_state()

    st.title("AI-Generated Image Detection")
    st.caption(
        "Pick a detection method, choose an image, apply one or more "
        "real-world-style transformations, and see how the detector responds."
    )

    method = sidebar.render_method_picker()
    image_choice = sidebar.render_image_source_picker()
    transform_pipeline = sidebar.render_transform_pipeline_controls()
    run_clicked = sidebar.render_run_button()

    # Live preview: show what the model will see as soon as an image + transforms
    # are chosen, without requiring a Run click.
    if image_choice is not None:
        image, _source_label = image_choice
        preview_image = apply_transform_pipeline(image, transform_pipeline)
        preview.render_transform_preview(image, preview_image, transform_pipeline)

    if run_clicked:
        if image_choice is None:
            st.sidebar.error("Upload an image first.")
        else:
            image, source_label = image_choice
            _run_and_store(image, transform_pipeline, source_label, method)

    pipeline_result = state.get_pipeline_result()
    if pipeline_result is None:
        st.info("Choose an image and transformation(s) in the sidebar, then click **Run Detection**.")
        return

    results.render_pipeline_results(pipeline_result)
    detection_chart.render_detection_section(pipeline_result.detection_result)
