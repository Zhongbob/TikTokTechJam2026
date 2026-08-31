"""AIDE AI-generated-image detector (running/inference stage).

`AIDEDetector` implements the shared `detector_common.ImageDetector` /
`shared_types.interfaces.EnsembleDetector` contract, wrapping AIDE
(shilinyan99/AIDE) — an OpenCLIP ConvNeXt semantic branch fused with a
high/low-frequency patch branch.
"""

from aide.detector import AIDEDetector

__all__ = ["AIDEDetector"]
