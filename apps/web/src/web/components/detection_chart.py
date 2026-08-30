from __future__ import annotations

import pandas as pd
import streamlit as st
from shared_types.detection import DetectionResult


def render_detection_section(result: DetectionResult) -> None:
    st.subheader("Detection result")

    if result.is_placeholder:
        st.warning(
            f"**PLACEHOLDER MODEL — not yet trained** (`{result.model_version}`). "
            "Scores below are deterministic-but-fake, generated from the image itself. "
            "Swap the real ensemble in `services/factory.py` once "
            "`packages/models/ensemble` is ready.",
            icon="🚧",
        )

    verdict_label = "🤖 AI-Generated" if result.verdict == "ai_generated" else "✅ Real"
    col1, col2 = st.columns(2)
    col1.metric("Verdict", verdict_label)
    col2.metric("AI-generated probability", f"{result.ai_generated_probability:.1%}")

    if len(result.member_results) > 1:
        st.caption("Per-model breakdown")
        chart_data = pd.DataFrame(
            {
                "AI-generated probability": [m.ai_generated_probability for m in result.member_results],
                "Confidence": [m.confidence for m in result.member_results],
            },
            index=[m.model_name for m in result.member_results],
        )
        st.bar_chart(chart_data["AI-generated probability"])
        st.dataframe(chart_data.style.format("{:.1%}"), use_container_width=True)
    elif result.member_results:
        member = result.member_results[0]
        st.caption(f"Model: `{member.model_name}` · confidence {member.confidence:.1%}")
