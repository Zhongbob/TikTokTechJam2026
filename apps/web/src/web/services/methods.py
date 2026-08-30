"""The detection strategies the demo can run an image through.

- ``normal_classifier``: feed the (transformed) image straight into the trained
  ``NormalClassifierDetector`` — no restoration step.
- ``transform_reversal``: undo the degradation with the autoencoder restorer
  first, then run the ensemble detector on the restored image.
"""

from __future__ import annotations

NORMAL_CLASSIFIER = "normal_classifier"
TRANSFORM_REVERSAL = "transform_reversal"

METHODS: tuple[str, ...] = (NORMAL_CLASSIFIER, TRANSFORM_REVERSAL)

METHOD_LABELS: dict[str, str] = {
    NORMAL_CLASSIFIER: "Normal Classifier",
    TRANSFORM_REVERSAL: "Transform Reversal",
}

METHOD_DESCRIPTIONS: dict[str, str] = {
    NORMAL_CLASSIFIER: (
        "Runs the trained classifier directly on the (transformed) image. "
        "No restoration step."
    ),
    TRANSFORM_REVERSAL: (
        "Restores the image with the autoencoder first, then runs the ensemble "
        "detector on the restored image. (Both stages are still placeholders.)"
    ),
}
