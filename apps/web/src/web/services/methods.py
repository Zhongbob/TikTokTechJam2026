"""The detection strategies the demo can run an image through.

- ``yolo``: feed the (transformed) image straight into the trained
  YOLO classifier — no restoration step.
- ``transform_reversal``: run the ensemble detector on the transformed image;
  its fusion sub-model additionally scores an autoencoder-restored version
  (shown to the user).
"""

from __future__ import annotations

YOLO_CLASSIFIER = "yolo"
TRANSFORM_REVERSAL = "transform_reversal"

METHODS: tuple[str, ...] = (YOLO_CLASSIFIER, TRANSFORM_REVERSAL)

METHOD_LABELS: dict[str, str] = {
    YOLO_CLASSIFIER: "YOLO Classifier",
    TRANSFORM_REVERSAL: "Ensemble (Transform Reversal)",
}

METHOD_DESCRIPTIONS: dict[str, str] = {
    YOLO_CLASSIFIER: (
        "Runs the trained YOLO classifier directly on the (transformed) image. "
        "No restoration step."
    ),
    TRANSFORM_REVERSAL: (
        "Runs the ensemble (fusion + CLIP + DINOv2 + YOLO + Swin) on the "
        "transformed image. The fusion sub-model also scores an "
        "autoencoder-restored version — shown in the results panel."
    ),
}
