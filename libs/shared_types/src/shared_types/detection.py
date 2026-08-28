from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EnsembleMemberResult:
    """Prediction from a single detector model inside the ensemble."""

    model_name: str
    ai_generated_probability: float  # 0.0-1.0
    confidence: float  # 0.0-1.0, model's self-reported certainty
    is_placeholder: bool = True


@dataclass(frozen=True)
class DetectionResult:
    """Aggregated ensemble verdict for one (restored) image."""

    verdict: Literal["real", "ai_generated"]
    ai_generated_probability: float  # 0.0-1.0, ensemble-aggregated
    member_results: tuple[EnsembleMemberResult, ...]
    is_placeholder: bool = True
    model_version: str = "placeholder-v0"
