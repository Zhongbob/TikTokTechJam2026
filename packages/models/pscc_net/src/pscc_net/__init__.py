"""PSCC-Net image-manipulation detection + localization (inference stage).

`PSCCNetDetector` implements the shared `detector_common.ImageDetector` /
`shared_types.interfaces.EnsembleDetector` contract. PSCC-Net (Progressive
Spatio-Channel Correlation Network, proteus1991/PSCC-Net, MIT) pairs an
HRNet-W18-small backbone with a progressive non-local localization head and a
binary detection head; it targets *locally tampered* images (splice / copy-move
/ inpainting), so it is a natural complement to a whole-image synthetic-image
detector inside a fusion model.

The model architecture is vendored (`_hrnet.py`, `_heads.py`, `_model.py`) — no
repo clone needed. Weights are the repo's three checkpoints, auto-downloaded by
`PSCCNetDetector.use_default()`.
"""

from pscc_net.detector import PSCCNetDetector
from pscc_net._model import PSCCNet, build_pscc_net, load_pscc_weights

__all__ = ["PSCCNetDetector", "PSCCNet", "build_pscc_net", "load_pscc_weights"]
