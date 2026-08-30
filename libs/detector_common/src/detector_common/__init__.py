"""Shared plumbing for single-model AI-generated-image detectors.

Each concrete detector package (``community_forensics``, ``patchcraft``,
``drct``, ``aide``, ...) subclasses :class:`ImageDetector` and implements
``_score(image) -> p(ai_generated)``. The base supplies the
``shared_types.interfaces.EnsembleDetector`` surface (``name`` /
``is_placeholder`` / ``predict``) and an ``evaluate()`` method identical in
shape to ``NormalClassifierDetector.evaluate()``.
"""

from detector_common.base import ImageDetector, resolve_device, save_confusion_matrix

__all__ = ["ImageDetector", "resolve_device", "save_confusion_matrix"]
