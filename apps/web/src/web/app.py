from __future__ import annotations

import streamlit as st

from web import state
from web.components import detection_chart, preview, results, sidebar
from web.services.dataset import (
    load_sample_manifest,
    load_sample_image,
    pick_random_sample,
    pick_random_transform_pipeline,
)
from web.services.pipeline import run_pipeline
from web.services.transforms import apply_transform_pipeline


def render_app() -> None:
    state.init_session_state()

    st.title("AI-Generated Image Detection")
    st.caption(
        "Upload an image, apply one or more real-world-style transformations, and see how the "
        "autoencoder + ensemble detector pipeline responds. "
        "**The restoration and detection models are placeholders** until the real "
        "models are trained — see the warnings on the results below."
    )

    image_choice = sidebar.render_image_source_picker()
    transform_pipeline = sidebar.render_transform_pipeline_controls()
    run_clicked = sidebar.render_run_button()
    randomize_clicked = sidebar.render_randomize_button()

    # Live preview: show what the model will see as soon as an image + transforms
    # are chosen, without requiring a Run click.
    if image_choice is not None:
        image, _source_label = image_choice
        preview_image = apply_transform_pipeline(image, transform_pipeline)
        preview.render_transform_preview(image, preview_image, transform_pipeline)

    if randomize_clicked:
        manifest = load_sample_manifest()
        if manifest:
            sample = pick_random_sample(manifest)
            image = load_sample_image(sample)
            random_pipeline = pick_random_transform_pipeline()
            result = run_pipeline(image, random_pipeline, f"sample:{sample.img_id}")
            state.set_pipeline_result(result)
        else:
            st.sidebar.warning("No sample dataset images available to randomize from.")
    elif run_clicked:
        if image_choice is None:
            st.sidebar.error("Choose or upload an image first.")
        else:
            image, source_label = image_choice
            result = run_pipeline(image, transform_pipeline, source_label)
            state.set_pipeline_result(result)

    pipeline_result = state.get_pipeline_result()
    if pipeline_result is None:
        st.info("Choose an image and transformation(s) in the sidebar, then click **Run Detection**.")
        return

    results.render_pipeline_results(pipeline_result)
    detection_chart.render_detection_section(pipeline_result.detection_result)
