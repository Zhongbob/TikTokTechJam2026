"""The "running" stage: inference-ready wrapper around a trained
NormalClassifierTrainer checkpoint.

Implements `shared_types.interfaces.EnsembleDetector` — the same "ready"
contract apps/web already consumes (see
apps/web/src/web/services/factory.py's SWAP POINT comments), so this class
can be dropped straight in as a real detector once a checkpoint exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from shared_types.detection import DetectionResult, EnsembleMemberResult
from shared_types.interfaces import EnsembleDetector


class NormalClassifierDetector(EnsembleDetector):
    """Extend/instantiate this once you have a trained checkpoint:

        from normal_classifier import NormalClassifierDetector

        detector = NormalClassifierDetector.from_checkpoint("normal_classifier.pt")
        result = detector.predict(some_pil_image)
    """

    name = "normal-classifier-yolo"
    is_placeholder = False

    def __init__(self, model: Any) -> None:
        self._model = model  # an ultralytics.YOLO instance

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "NormalClassifierDetector":
        from ultralytics import YOLO

        return cls(YOLO(str(path)))

    def predict(self, image: Image.Image) -> DetectionResult:
        results = self._model.predict(image, verbose=False)
        probs = results[0].probs
        names: dict[int, str] = results[0].names
        name_to_index = {class_name: index for index, class_name in names.items()}
        if "ai_generated" not in name_to_index:
            raise ValueError(
                f"Loaded model's classes {list(names.values())} don't include 'ai_generated' — "
                "was it trained with NormalClassifierTrainer's class-folder layout?"
            )

        ai_generated_probability = float(probs.data[name_to_index["ai_generated"]])
        member = EnsembleMemberResult(
            model_name=self.name,
            ai_generated_probability=ai_generated_probability,
            confidence=float(probs.top1conf),
            is_placeholder=False,
        )
        return DetectionResult(
            verdict="ai_generated" if ai_generated_probability >= 0.5 else "real",
            ai_generated_probability=ai_generated_probability,
            member_results=(member,),
            is_placeholder=False,
            model_version=self.name,
        )
