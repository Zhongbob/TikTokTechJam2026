"""PatchCraft AI-generated-image detector (running/inference stage).

`PatchCraftDetector` implements the shared `detector_common.ImageDetector` /
`shared_types.interfaces.EnsembleDetector` contract. It reproduces PatchCraft's
"Smash & Reconstruct" texture-residual preprocessing (arXiv:2311.12397) and
feeds the result to an EfficientNet-B4 classifier.
"""

from patchcraft.detector import PatchCraftDetector

__all__ = ["PatchCraftDetector"]
