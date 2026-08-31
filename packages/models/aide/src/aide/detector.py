"""Inference wrapper around the AIDE detector (shilinyan99/AIDE).

AIDE = "AI-generated Image DEtector": two branches fused by an MLP head —

  * a **semantic** branch: an OpenCLIP ConvNeXt-Base image encoder;
  * a **frequency** branch: the highest- and lowest-frequency image patches
    (selected via a DCT energy score) passed through a small CNN.

The official repo ships ProGAN / GenImage / SDv1.4 trained variants.

`torch` / `open_clip` / `numpy` are imported lazily.

⚠️ FIDELITY: AIDE's fusion head and the exact frequency-branch CNN live in the
official repo's model code, which is not reproduced here. Use one of:

  * `AIDEDetector.from_module(model, preprocess=...)` — you build the official
    `AIDE(...)` module and load its checkpoint yourself, then hand it over;
  * `AIDEDetector.from_checkpoint(path, arch_factory=lambda: AIDE(...))` — pass
    a factory for the official architecture and this loads the state dict.

`use_default()` raises with these instructions until a checkpoint + arch are
wired in. The DCT patch selection below matches the paper and can be reused.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from detector_common import ImageDetector, resolve_device
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = SCRIPT_DIR.parent / "weights" / "aide_sdv14.pth"

SEMANTIC_SIZE = 224
PATCH_SIZE = 32
N_FREQ_PATCHES = 4  # top-K highest + bottom-K lowest frequency patches
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class AIDEDetector(ImageDetector):
    """AIDE two-branch model, wrapped as an `ImageDetector`.

        # with the official architecture available as `AIDE`
        detector = AIDEDetector.from_checkpoint("aide_sdv14.pth", arch_factory=lambda: AIDE(...))
        result = detector.predict(pil_image)
    """

    name = "aide"
    is_placeholder = False

    def __init__(
        self,
        model: Any,
        *,
        preprocess: Callable[[Image.Image], Any] | None = None,
        device: str = "auto",
        positive_index: int = 1,
    ) -> None:
        import torch

        self.device = resolve_device(device)
        self._model = model.to(self.device).eval()
        self._preprocess = preprocess or self._default_preprocess
        self._torch = torch
        self._positive_index = positive_index

    # --- construction --------------------------------------------------

    @classmethod
    def from_module(
        cls,
        model: Any,
        *,
        preprocess: Callable[[Image.Image], Any] | None = None,
        device: str = "auto",
        positive_index: int = 1,
    ) -> "AIDEDetector":
        """Wrap an already-built + already-loaded AIDE ``nn.Module``."""
        return cls(model, preprocess=preprocess, device=device, positive_index=positive_index)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        arch_factory: Callable[[], Any],
        preprocess: Callable[[Image.Image], Any] | None = None,
        device: str = "auto",
        positive_index: int = 1,
    ) -> "AIDEDetector":
        import torch

        model = arch_factory()
        state = torch.load(str(path), map_location="cpu")
        state = state.get("model", state.get("state_dict", state))
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"[aide] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys")
        return cls(model, preprocess=preprocess, device=device, positive_index=positive_index)

    @classmethod
    def use_default(cls, *, device: str = "auto") -> "AIDEDetector":
        raise FileNotFoundError(
            "AIDE needs both a checkpoint and its architecture. Clone shilinyan99/AIDE, then:\n"
            "  from aide_repo.models import AIDE\n"
            "  AIDEDetector.from_checkpoint('aide_sdv14.pth', arch_factory=lambda: AIDE(...))\n"
            f"(place the checkpoint at {DEFAULT_CHECKPOINT} and extend use_default() once wired.)"
        )

    # --- preprocessing --------------------------------------------

    def _default_preprocess(self, image: Image.Image) -> dict[str, Any]:
        """Produce {'semantic': (1,3,H,W), 'freq': (1,2*K,3,P,P)} tensors.

        The official model may expect a different packing — override via the
        `preprocess=` argument if so.
        """
        import numpy as np
        import torch

        rgb = image.convert("RGB")

        # semantic branch: resize + centre-crop to 224, CLIP normalisation
        s = rgb.resize((SEMANTIC_SIZE, SEMANTIC_SIZE), Image.BICUBIC)
        s_arr = np.asarray(s, dtype=np.float32) / 255.0
        s_arr = (s_arr - np.asarray(_CLIP_MEAN)) / np.asarray(_CLIP_STD)
        semantic = torch.from_numpy(s_arr).permute(2, 0, 1).unsqueeze(0).float()

        # frequency branch: DCT-energy-ranked patches
        freq_patches = _select_frequency_patches(rgb, PATCH_SIZE, N_FREQ_PATCHES)
        freq = torch.from_numpy(freq_patches).float().unsqueeze(0)  # (1, 2K, 3, P, P)

        return {"semantic": semantic.to(self.device), "freq": freq.to(self.device)}

    # --- scoring ---------------------------------------------------

    def _score(self, image: Image.Image) -> float:
        torch = self._torch
        inputs = self._preprocess(image)
        with torch.no_grad():
            if isinstance(inputs, dict):
                out = self._model(**inputs)
            else:
                out = self._model(inputs)
            logits = (out.logits if hasattr(out, "logits") else out).float().squeeze(0)
        if logits.ndim == 0 or logits.shape[-1] == 1:
            return float(torch.sigmoid(logits.reshape(())).item())
        return float(torch.softmax(logits, dim=-1)[self._positive_index].item())


def _select_frequency_patches(image: Image.Image, patch: int, k: int) -> "Any":
    """Return the `k` highest- and `k` lowest-frequency patches, stacked as a
    (2k, 3, patch, patch) float32 array in [0, 1]. Frequency score = energy of
    the DCT coefficients outside the top-left (low-freq) quadrant."""
    import numpy as np

    side = patch * max(8, k * 2)
    arr = np.asarray(image.resize((side, side), Image.BILINEAR), dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)

    coords, scores = [], []
    for y in range(0, side - patch + 1, patch):
        for x in range(0, side - patch + 1, patch):
            block = gray[y:y + patch, x:x + patch]
            coeffs = _dct2(block)
            lo = patch // 4
            coeffs[:lo, :lo] = 0.0
            coords.append((y, x))
            scores.append(float(np.abs(coeffs).sum()))

    order = np.argsort(scores)
    picks = [coords[i] for i in order[-k:]] + [coords[i] for i in order[:k]]
    out = np.stack([arr[y:y + patch, x:x + patch].transpose(2, 0, 1) for y, x in picks])
    return out.astype(np.float32)


def _dct2(block: "Any") -> "Any":
    """2-D DCT-II via matrix multiply (no SciPy dependency)."""
    import numpy as np

    n = block.shape[0]
    k = np.arange(n)
    basis = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    basis[0, :] *= 1 / np.sqrt(2)
    basis *= np.sqrt(2 / n)
    return basis @ block @ basis.T
