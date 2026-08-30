"""Inference wrapper around the PatchCraft detector (arXiv:2311.12397).

PatchCraft = "Smash & Reconstruct": split the image into small patches, rank
them by *texture diversity*, rebuild one image from the richest-texture patches
and one from the poorest, run both through a bank of high-pass (SRM-style)
filters, and feed the (rich - poor) residual to an EfficientNet-B4 classifier.
The intuition: real and generated images differ more in fine texture statistics
than in content.

`torch` / `timm` / `numpy` are imported lazily.

⚠️ FIDELITY: the patch-ranking + reconstruction below follows the paper, but
the official repo's exact 30-filter SRM bank and any train-time normalisation
are NOT reproduced here (a compact high-pass bank is used instead). Validate
against the official release before trusting absolute scores; wire the real
filter bank into `_HIGH_PASS_KERNELS` and adjust `PATCH_SIZE` / input size to
match the checkpoint you load.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from detector_common import ImageDetector, resolve_device
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR.parent / "weights" / "patchcraft_efficientnet_b4.pth"

#: Side length (px) of the square patches ranked by texture diversity.
PATCH_SIZE = 32
#: Reconstructed rich/poor images are this many patches per side.
GRID = 8
INPUT_SIZE = PATCH_SIZE * GRID  # 256

# A small high-pass bank standing in for PatchCraft's 30 SRM filters.
_HIGH_PASS_KERNELS = (
    ((0, -1, 0), (-1, 4, -1), (0, -1, 0)),
    ((-1, -1, -1), (-1, 8, -1), (-1, -1, -1)),
    ((1, -2, 1), (-2, 4, -2), (1, -2, 1)),
)


class PatchCraftDetector(ImageDetector):
    """PatchCraft EfficientNet-B4, wrapped as an `ImageDetector`.

        detector = PatchCraftDetector.use_default()
        result = detector.predict(pil_image)
    """

    name = "patchcraft-efficientnet-b4"
    is_placeholder = False

    def __init__(
        self,
        model: Any,
        *,
        device: str = "auto",
        positive_index: int = 1,
    ) -> None:
        import torch

        self.device = resolve_device(device)
        self._model = model.to(self.device).eval()
        self._torch = torch
        self._positive_index = positive_index

    # --- construction --------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "auto",
        backbone: str = "tf_efficientnet_b4_ns",
        num_classes: int = 2,
        positive_index: int = 1,
    ) -> "PatchCraftDetector":
        import timm
        import torch

        model = timm.create_model(backbone, pretrained=False, num_classes=num_classes)
        state = torch.load(str(path), map_location="cpu")
        state = state.get("model", state.get("state_dict", state))
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[patchcraft] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
        return cls(model, device=device, positive_index=positive_index)

    @classmethod
    def use_default(cls, *, device: str = "auto") -> "PatchCraftDetector":
        if not DEFAULT_CHECKPOINT.is_file():
            raise FileNotFoundError(
                f"PatchCraft checkpoint not found at {DEFAULT_CHECKPOINT}. Download the "
                "EfficientNet-B4 weights from the official PatchCraft release "
                "(arXiv:2311.12397) and place them there, or call from_checkpoint(path)."
            )
        return cls.from_checkpoint(DEFAULT_CHECKPOINT, device=device)

    # --- Smash & Reconstruct preprocessing ---------------------------

    @staticmethod
    def _texture_diversity(patch: "Any") -> float:
        """Sum of |neighbour differences| in 4 directions — higher = richer texture."""
        import numpy as np

        p = patch.astype(np.float32)
        h = np.abs(p[:, 1:] - p[:, :-1]).sum()
        v = np.abs(p[1:, :] - p[:-1, :]).sum()
        d1 = np.abs(p[1:, 1:] - p[:-1, :-1]).sum()
        d2 = np.abs(p[1:, :-1] - p[:-1, 1:]).sum()
        return float(h + v + d1 + d2)

    def _smash_and_reconstruct(self, image: Image.Image) -> "Any":
        """Return the (rich - poor) high-pass residual as a float32 HxWx3 array."""
        import numpy as np

        side = INPUT_SIZE
        arr = np.asarray(image.convert("RGB").resize((side, side), Image.BILINEAR), dtype=np.uint8)
        gray = arr.mean(axis=2)

        patches, scores = [], []
        for gy in range(GRID):
            for gx in range(GRID):
                y, x = gy * PATCH_SIZE, gx * PATCH_SIZE
                patches.append(arr[y:y + PATCH_SIZE, x:x + PATCH_SIZE])
                scores.append(self._texture_diversity(gray[y:y + PATCH_SIZE, x:x + PATCH_SIZE]))

        order = np.argsort(scores)  # ascending texture
        n = GRID * GRID
        rich = self._tile([patches[i] for i in order[::-1]])
        poor = self._tile([patches[i] for i in order])

        residual = self._high_pass(rich) - self._high_pass(poor)
        return residual

    @staticmethod
    def _tile(patches: list) -> "Any":
        import numpy as np

        rows = [np.concatenate(patches[r * GRID:(r + 1) * GRID], axis=1) for r in range(GRID)]
        return np.concatenate(rows, axis=0).astype(np.float32)

    @staticmethod
    def _high_pass(arr: "Any") -> "Any":
        """Average of a few high-pass filter responses (per channel)."""
        import numpy as np

        x = arr / 255.0
        acc = np.zeros_like(x)
        for kernel in _HIGH_PASS_KERNELS:
            k = np.asarray(kernel, dtype=np.float32)
            for c in range(3):
                acc[..., c] += _conv2d_same(x[..., c], k)
        return acc / len(_HIGH_PASS_KERNELS)

    # --- scoring ---------------------------------------------------

    def _score(self, image: Image.Image) -> float:
        torch = self._torch
        residual = self._smash_and_reconstruct(image)  # HxWx3 float32
        tensor = torch.from_numpy(residual).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            logits = self._model(tensor).float().squeeze(0)
        if logits.ndim == 0 or logits.shape[-1] == 1:
            return float(torch.sigmoid(logits.reshape(())).item())
        return float(torch.softmax(logits, dim=-1)[self._positive_index].item())


def _conv2d_same(image: "Any", kernel: "Any") -> "Any":
    """Tiny 'same' 2-D convolution with edge padding (numpy, no SciPy dep)."""
    import numpy as np

    kh, kw = kernel.shape
    padded = np.pad(image, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode="edge")
    out = np.zeros_like(image)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i, j] * padded[i:i + image.shape[0], j:j + image.shape[1]]
    return out
