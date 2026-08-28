"""One-off dev script: generates the placeholder sample dataset bundled with
the web app (apps/web/src/web/assets/sample_dataset/).

These are procedurally-drawn synthetic images (no external licensing
concerns), not real photos or real AI-generated images. Swap this dataset
for a real curated one by replacing manifest.json + images/ on disk — the
DatasetSample schema and services/dataset.py loader do not need to change.

Run manually (not at app startup):
    uv run python apps/web/scripts/generate_sample_dataset.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "src" / "web" / "assets" / "sample_dataset"
_IMAGES_DIR = _OUTPUT_DIR / "images"
_IMAGE_SIZE = (256, 256)


def _draw_real_sample(seed: int) -> Image.Image:
    """Organic-looking placeholder: soft gradient + irregular blobs, meant to
    stand in for an unaltered photo."""
    rng = random.Random(seed)
    image = Image.new("RGB", _IMAGE_SIZE)
    top_color = tuple(rng.randint(60, 200) for _ in range(3))
    bottom_color = tuple(rng.randint(20, 160) for _ in range(3))
    for y in range(_IMAGE_SIZE[1]):
        t = y / (_IMAGE_SIZE[1] - 1)
        row_color = tuple(round(top_color[c] * (1 - t) + bottom_color[c] * t) for c in range(3))
        for x in range(_IMAGE_SIZE[0]):
            image.putpixel((x, y), row_color)

    draw = ImageDraw.Draw(image)
    for _ in range(6):
        cx, cy = rng.randint(0, _IMAGE_SIZE[0]), rng.randint(0, _IMAGE_SIZE[1])
        radius = rng.randint(15, 45)
        color = tuple(rng.randint(0, 255) for _ in range(3))
        draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=color, outline=None)
    return image


def _draw_ai_generated_sample(seed: int) -> Image.Image:
    """Artificial-looking placeholder: crisp geometric shapes on a flat
    background, meant to stand in for a synthetic/AI-generated image."""
    rng = random.Random(seed)
    background = tuple(rng.randint(0, 255) for _ in range(3))
    image = Image.new("RGB", _IMAGE_SIZE, color=background)
    draw = ImageDraw.Draw(image)
    for _ in range(5):
        shape_kind = rng.choice(["rectangle", "polygon", "line"])
        color = tuple(rng.randint(0, 255) for _ in range(3))
        if shape_kind == "rectangle":
            x0, y0 = rng.randint(0, 180), rng.randint(0, 180)
            x1, y1 = x0 + rng.randint(20, 76), y0 + rng.randint(20, 76)
            draw.rectangle((x0, y0, x1, y1), outline=color, width=rng.randint(2, 6))
        elif shape_kind == "polygon":
            points = [(rng.randint(0, 256), rng.randint(0, 256)) for _ in range(rng.randint(3, 5))]
            draw.polygon(points, outline=color, width=rng.randint(2, 4))
        else:
            draw.line(
                (rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256), rng.randint(0, 256)),
                fill=color,
                width=rng.randint(2, 6),
            )
    return image


def main() -> None:
    _IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    manifest_entries = []

    for i in range(1, 4):
        img_id = f"real_{i:03d}"
        file_name = f"{img_id}.jpg"
        _draw_real_sample(seed=i).save(_IMAGES_DIR / file_name, format="JPEG", quality=95)
        manifest_entries.append(
            {
                "img_id": img_id,
                "file_name": file_name,
                "label_name": "real",
                "binary_aigc_label": 0,
                "sid_label": 0,
                "description": "Placeholder real-photo sample (procedurally generated).",
            }
        )

    for i in range(1, 4):
        img_id = f"ai_{i:03d}"
        file_name = f"{img_id}.jpg"
        _draw_ai_generated_sample(seed=100 + i).save(_IMAGES_DIR / file_name, format="JPEG", quality=95)
        manifest_entries.append(
            {
                "img_id": img_id,
                "file_name": file_name,
                "label_name": "ai_generated",
                "binary_aigc_label": 1,
                "sid_label": 0,
                "description": "Placeholder AI-generated sample (procedurally generated).",
            }
        )

    with open(_OUTPUT_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_entries, f, indent=2)

    print(f"Wrote {len(manifest_entries)} sample images + manifest to {_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
