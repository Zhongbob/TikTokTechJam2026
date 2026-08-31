"""Community-Forensics AI-generated-image detector (running/inference stage).

`CommunityForensicsDetector` implements the shared `detector_common.ImageDetector`
/ `shared_types.interfaces.EnsembleDetector` contract, wrapping the ViT model
from JeongsooP/Community-Forensics (HF mirror
``buildborderless/CommunityForensics-DeepfakeDet-ViT``).
"""

from community_forensics.detector import CommunityForensicsDetector

__all__ = ["CommunityForensicsDetector"]
