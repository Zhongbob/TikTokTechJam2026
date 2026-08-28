from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from shared_types.augmentation import SourceMetadata


@dataclass(frozen=True)
class DatasetSample:
    """One entry in a (placeholder or real) test-image dataset manifest."""

    img_id: str
    file_name: str  # relative to the dataset's images/ directory
    label_name: Literal["real", "ai_generated"]
    binary_aigc_label: int  # 0 = real, 1 = ai_generated
    sid_label: int = 0  # unused for the placeholder set, kept for schema parity
    description: str = ""

    def as_source_metadata(self) -> SourceMetadata:
        return SourceMetadata(
            img_id=self.img_id,
            sid_label=self.sid_label,
            label_name=self.label_name,
            binary_aigc_label=self.binary_aigc_label,
        )
