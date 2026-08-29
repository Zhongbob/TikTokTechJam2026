from __future__ import annotations

import streamlit as st
from PIL import Image
from shared_types.transforms import TRANSFORM_DISPLAY_NAMES, TransformPipeline


def render_transform_preview(original: Image.Image, transformed: Image.Image, pipeline: TransformPipeline) -> None:
    """Shows what the image will look like after the selected transform chain,
    before it's fed into the (placeholder) restoration/detection models.
    """
    st.subheader("Preview")
    if pipeline:
        chain = " → ".join(TRANSFORM_DISPLAY_NAMES[spec.transform_type] for spec in pipeline)
        st.caption(f"This is what the model will see after: **{chain}**")
    else:
        st.caption("No transformations selected — the model will see the original image as-is.")

    columns = st.columns(2)
    with columns[0]:
        st.image(original, caption="Original", use_container_width=True)
    with columns[1]:
        st.image(transformed, caption="Preview after transformation(s)", use_container_width=True)
