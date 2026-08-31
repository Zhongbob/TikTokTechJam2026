from __future__ import annotations

import streamlit as st
from shared_types.transforms import TRANSFORM_DISPLAY_NAMES

from web.services.methods import METHOD_LABELS, TRANSFORM_REVERSAL
from web.services.pipeline import PipelineResult


def describe_pipeline(result: PipelineResult) -> str:
    if not result.transform_pipeline:
        return "none (original image)"
    return " → ".join(TRANSFORM_DISPLAY_NAMES[spec.transform_type] for spec in result.transform_pipeline)


def render_pipeline_results(result: PipelineResult) -> None:
    st.subheader("Pipeline results")

    pipeline_description = describe_pipeline(result)
    st.caption(
        f"Method: **{METHOD_LABELS[result.method]}** · "
        f"Source: `{result.augmentation_record.source}` · "
        f"Transforms applied: **{pipeline_description}** · "
        f"Parameters: `{result.augmentation_record.parameters}`"
    )

    if result.method == TRANSFORM_REVERSAL:
        columns = st.columns(3)
        with columns[0]:
            st.image(result.original_image, caption="1. Original", use_container_width=True)
        with columns[1]:
            st.image(
                result.transformed_image,
                caption=f"2. Transformed ({pipeline_description})",
                use_container_width=True,
            )
        with columns[2]:
            restored_caption = "3. Autoencoder-restored (what the fusion sub-model scores)"
            if result.restorer_is_placeholder:
                restored_caption += " ⚠️ placeholder"
            st.image(result.restored_image, caption=restored_caption, use_container_width=True)

        if result.restorer_is_placeholder:
            st.info(
                "The trained autoencoder wasn't available in this environment, so the "
                "restored panel shows a placeholder smoothing filter. The ensemble's "
                "detection result is unaffected.",
                icon="⚠️",
            )
        return

    # YOLO classifier: no restoration stage.
    columns = st.columns(2)
    with columns[0]:
        st.image(result.original_image, caption="1. Original", use_container_width=True)
    with columns[1]:
        st.image(
            result.transformed_image,
            caption=f"2. Transformed ({pipeline_description})",
            use_container_width=True,
        )
    st.caption("The **Normal Classifier** runs directly on the transformed image — no restoration step.")
