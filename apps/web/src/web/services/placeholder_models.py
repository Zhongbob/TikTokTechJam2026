"""Stand-in implementations of the AutoencoderRestorer / EnsembleDetector
interfaces, used until the real models in packages/models/* are ready.

Swap these out in services/factory.py once real implementations exist —
nothing else in the app needs to change, since both sides of the swap
satisfy the same shared_types.interfaces Protocols.
"""

from __future__ import annotations

import hashlib
import random

from PIL import Image, ImageFilter
from shared_types.detection import DetectionResult, EnsembleMemberResult

_MEMBER_MODEL_NAMES = (
    "CNNDetector-A",
    "ViTDetector-B",
    "FrequencyArtifactNet",
    "DiffusionFingerprintNet",
)


class DummyAutoencoderRestorer:
    """Cosmetic stand-in for the real autoencoder restoration model.

    Applies a smoothing filter so the "restored" panel is visibly distinct
    from the transformed input, rather than a no-op passthrough.
    """

    name = "Placeholder Autoencoder (SMOOTH_MORE filter)"
    is_placeholder = True

    def restore(self, image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.SMOOTH_MORE)


class DummyEnsembleDetector:
    """Fake ensemble detector with deterministic-per-image scores.

    Scores are seeded from a hash of the image bytes so the same image
    always yields the same fake verdict (stable demo UX, testable).
    """

    name = "Placeholder Ensemble (untrained)"
    is_placeholder = True

    def predict(self, image: Image.Image) -> DetectionResult:
        image_bytes = image.convert("RGB").tobytes()
        seed = int(hashlib.sha256(image_bytes).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)

        member_results = tuple(
            EnsembleMemberResult(
                model_name=model_name,
                ai_generated_probability=rng.random(),
                confidence=rng.uniform(0.5, 1.0),
                is_placeholder=True,
            )
            for model_name in _MEMBER_MODEL_NAMES
        )
        aggregate_probability = sum(m.ai_generated_probability for m in member_results) / len(
            member_results
        )
        verdict = "ai_generated" if aggregate_probability >= 0.5 else "real"

        return DetectionResult(
            verdict=verdict,
            ai_generated_probability=aggregate_probability,
            member_results=member_results,
            is_placeholder=True,
            model_version="placeholder-v0",
        )
