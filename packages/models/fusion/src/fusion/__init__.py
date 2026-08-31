"""Fusion model — Community-Forensics + OpenSDI combined into one detector.

`FusionDetector` (inference) implements the shared
`detector_common.ImageDetector` / `shared_types.interfaces.EnsembleDetector`
contract. Default decider: ``max`` of the members' p(ai) vs a threshold — fake
if *either* the whole-image synthetic detector (Community-Forensics) or the
diffusion-inpainting tamper localizer (OpenSDI / MaskCLIP) fires.

`FusionTrainer` (training) is a **placeholder** for a future learned
meta-classifier over member scores; every method raises ``NotImplementedError``.
"""

from fusion.detector import FusionDetector
from fusion.trainer import FusionTrainer

__all__ = ["FusionDetector", "FusionTrainer"]
