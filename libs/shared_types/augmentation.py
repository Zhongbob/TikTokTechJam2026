"""Common types shared by augmentation, dataset, and retrieval modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, TypedDict

import numpy as np
from PIL import Image

ImageInput: TypeAlias = str | Path | Image.Image | np.ndarray


@dataclass(frozen=True)
class AugmentationRecord:
    """Description of the transformations applied to one image."""

    source: str
    transform: str
    parameters: dict[str, Any]


class SourceMetadata(TypedDict, total=False):
    """Optional source information copied into generated manifest entries."""

    img_id: str
    sid_label: int
    label_name: str
    binary_aigc_label: int

