from __future__ import annotations

import streamlit as st
from PIL import Image
from shared_types.transforms import (
    BLUR_SIGMAS,
    CENTER_CROP_FRACTION,
    COLOR_JITTER_RANGE,
    JPEG_QUALITIES,
    NOISE_SIGMAS,
    RESIZE_SCALES,
    TRANSFORM_DESCRIPTIONS,
    TRANSFORM_DISPLAY_NAMES,
    CenterCropParams,
    ColorJitterParams,
    GaussianBlurParams,
    GaussianNoiseParams,
    JpegCompressionParams,
    ResizeParams,
    TransformPipeline,
    TransformSpec,
    TransformType,
)

from web.services.dataset import load_sample_manifest, load_sample_image
from web.services.methods import METHOD_DESCRIPTIONS, METHOD_LABELS, METHODS


def render_method_picker() -> str:
    """Lets the user choose between the 'Normal Classifier' and 'Transform
    Reversal' detection methods. Returns the selected method key."""
    st.sidebar.subheader("Detection method")
    method = st.sidebar.radio(
        "Detection method",
        METHODS,
        format_func=lambda m: METHOD_LABELS[m],
        label_visibility="collapsed",
    )
    st.sidebar.caption(METHOD_DESCRIPTIONS[method])
    return method


def render_image_source_picker() -> tuple[Image.Image, str] | None:
    """Renders the upload-vs-sample-dataset picker.

    Returns (image, source_label) for whichever source currently has a
    valid image selected, or None if nothing is selected yet.
    """
    st.sidebar.subheader("1. Choose an image")
    source_mode = st.sidebar.radio(
        "Image source", ["Upload your own", "Sample dataset"], label_visibility="collapsed"
    )

    if source_mode == "Upload your own":
        uploaded_file = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is None:
            return None
        image = Image.open(uploaded_file).convert("RGB")
        return image, f"upload:{uploaded_file.name}"

    manifest = load_sample_manifest()
    if not manifest:
        st.sidebar.warning("No sample dataset images available.")
        return None
    labels = [f"{s.img_id} ({s.label_name})" for s in manifest]
    selected_index = st.sidebar.selectbox(
        "Sample image", range(len(manifest)), format_func=lambda i: labels[i]
    )
    sample = manifest[selected_index]
    image = load_sample_image(sample)
    return image, f"sample:{sample.img_id}"


def _render_transform_params(transform_type: TransformType) -> TransformSpec:
    """Renders the parameter widget(s) for one transform type and returns its spec.

    Each widget gets a key scoped to the transform type so multiple steps'
    widgets can coexist in the same render without colliding.
    """
    st.sidebar.caption(TRANSFORM_DESCRIPTIONS[transform_type])
    match transform_type:
        case TransformType.JPEG_COMPRESSION:
            quality = st.sidebar.select_slider(
                "JPEG quality", options=JPEG_QUALITIES, value=JPEG_QUALITIES[0], key="param_jpeg_quality"
            )
            params = JpegCompressionParams(quality=quality)
        case TransformType.GAUSSIAN_BLUR:
            sigma = st.sidebar.select_slider(
                "Kernel sigma (σ)", options=BLUR_SIGMAS, value=BLUR_SIGMAS[0], key="param_blur_sigma"
            )
            params = GaussianBlurParams(sigma=sigma)
        case TransformType.RESIZE:
            scale = st.sidebar.select_slider(
                "Downscale factor", options=RESIZE_SCALES, value=RESIZE_SCALES[0], key="param_resize_scale"
            )
            params = ResizeParams(scale=scale)
        case TransformType.GAUSSIAN_NOISE:
            sigma = st.sidebar.select_slider(
                "Noise sigma (σ)", options=NOISE_SIGMAS, value=NOISE_SIGMAS[0], key="param_noise_sigma"
            )
            params = GaussianNoiseParams(sigma=sigma)
        case TransformType.COLOR_JITTER:
            factor = st.sidebar.slider(
                "Jitter factor (brightness/contrast/saturation)",
                min_value=COLOR_JITTER_RANGE[0],
                max_value=COLOR_JITTER_RANGE[1],
                value=COLOR_JITTER_RANGE[1],
                step=0.05,
                key="param_jitter_factor",
            )
            params = ColorJitterParams(factor=factor)
        case TransformType.CENTER_CROP:
            st.sidebar.caption(f"Fixed crop: {int(CENTER_CROP_FRACTION * 100)}% of the image, centered.")
            params = CenterCropParams(crop_fraction=CENTER_CROP_FRACTION)
        case _:
            raise ValueError(f"Unhandled transform type: {transform_type}")

    return TransformSpec(transform_type=transform_type, params=params)


def render_transform_pipeline_controls() -> TransformPipeline:
    """Lets the user build an ordered chain of one or more transform steps.

    Steps run in the order they were added (top to bottom), so e.g. "Blur
    then Crop" gives a different result than "Crop then Blur".
    """
    st.sidebar.subheader("2. Apply transformation(s)")
    selected_types = st.sidebar.multiselect(
        "Transformations to apply, in order",
        list(TransformType),
        format_func=lambda t: TRANSFORM_DISPLAY_NAMES[t],
        help="Add one or more steps — each is applied in the order shown here.",
    )

    if not selected_types:
        st.sidebar.caption("No transformations selected — the original image will be used as-is.")
        return []

    pipeline: TransformPipeline = []
    for step_number, transform_type in enumerate(selected_types, start=1):
        st.sidebar.markdown(f"**Step {step_number}: {TRANSFORM_DISPLAY_NAMES[transform_type]}**")
        pipeline.append(_render_transform_params(transform_type))

    return pipeline


def render_run_button() -> bool:
    return st.sidebar.button("Run Detection", type="primary", use_container_width=True)


def render_randomize_button() -> bool:
    st.sidebar.subheader("3. Or, skip the setup")
    return st.sidebar.button("🎲 Randomize (sample image + transform)", use_container_width=True)
