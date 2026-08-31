"""Inference wrapper around PSCC-Net (proteus1991/PSCC-Net).

PSCC-Net = HRNet-W18-small feature extractor + a progressive non-local
**localization** head (per-pixel manipulation mask) + a binary **detection**
head ("manipulated vs authentic"). It is trained on splicing / copy-move /
inpainting (removal) — i.e. *locally* tampered images — so it complements a
whole-image synthetic-image detector rather than replacing one.

    detector = PSCCNetDetector.use_default()          # downloads 3 checkpoints
    result   = detector.predict(pil_image)            # DetectionResult
    mask     = detector.predict_mask(pil_image)       # H x W float array, [0, 1]
    metrics  = detector.evaluate(val_samples, generate_confusion_matrix=True)

Scoring (``score_mode``):
    * ``"label"`` (default) — the detection head's ``softmax(logit)[fake]``.
      PSCC-Net has a real classification head, so this is a genuine probability.
    * ``"mask"``  — reduce the localization mask to a scalar (``mask_reduce``:
      ``"mean"`` = fraction flagged, ``"max"`` = strongest region, or a float
      quantile). Useful when the detection head is miscalibrated on your data.
    * ``"hybrid"`` — ``max(p_label, mask_max)``: fire if *either* signal fires.

``flip=True`` inverts the final score if a labelled eval comes out reversed.

Preprocessing matches PSCC-Net's ``TestData``: RGB, resize to ``image_size`` and
scale to ``[0, 1]`` — **no ImageNet normalization**. ``image_size`` also drives
the head feature grid, so keep it at 256 unless you know why.

`torch` / `torchvision` are imported lazily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from detector_common import ImageDetector, resolve_device
from PIL import Image

from pscc_net._model import DEFAULT_CROP_SIZE, PSCCNet, build_pscc_net

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_DIR = SCRIPT_DIR.parent / "weights" / "pscc_net"

DEFAULT_INPUT_SIZE = 256

#: The three released checkpoints, straight from the repo (they are plain files
#: committed to git, ~15 MB total — no HF mirror).
_WEIGHT_URLS = {
    "HRNet.pth":
        "https://raw.githubusercontent.com/proteus1991/PSCC-Net/main/checkpoint/HRNet_checkpoint/HRNet.pth",
    "NLCDetection.pth":
        "https://raw.githubusercontent.com/proteus1991/PSCC-Net/main/checkpoint/NLCDetection_checkpoint/NLCDetection.pth",
    "DetectionHead.pth":
        "https://raw.githubusercontent.com/proteus1991/PSCC-Net/main/checkpoint/DetectionHead_checkpoint/DetectionHead.pth",
}

_SCORE_MODES = {"label", "mask", "hybrid"}


def _find_checkpoints(directory: Path) -> tuple[Path, Path, Path]:
    """Locate ``HRNet.pth`` / ``NLCDetection.pth`` / ``DetectionHead.pth`` under
    ``directory`` — accepts either a flat layout or the repo's nested
    ``<name>_checkpoint/<name>.pth`` layout."""
    directory = Path(directory)
    resolved: list[Path] = []
    for name in ("HRNet.pth", "NLCDetection.pth", "DetectionHead.pth"):
        stem = name[:-4]
        for candidate in (directory / name,
                          directory / f"{stem}_checkpoint" / name,
                          directory / stem / name):
            if candidate.is_file():
                resolved.append(candidate)
                break
        else:
            raise FileNotFoundError(
                f"{name} not found under {directory} (looked for a flat file and "
                f"a {stem}_checkpoint/ subdir)."
            )
    return resolved[0], resolved[1], resolved[2]


def _ensure_default_weights(weights_dir: Path) -> tuple[Path, Path, Path]:
    """Download the three PSCC-Net checkpoints into ``weights_dir`` if missing."""
    import torch

    weights_dir.mkdir(parents=True, exist_ok=True)
    for name, url in _WEIGHT_URLS.items():
        target = weights_dir / name
        if target.is_file():
            continue
        print(f"[pscc-net] downloading {name} -> {target}")
        torch.hub.download_url_to_file(url, str(target))
    return _find_checkpoints(weights_dir)


class PSCCNetDetector(ImageDetector):
    """PSCC-Net image-manipulation detector, wrapped as an `ImageDetector`."""

    name = "pscc-net"
    is_placeholder = False

    def __init__(
        self,
        model: PSCCNet,
        *,
        device: str = "auto",
        image_size: int = DEFAULT_INPUT_SIZE,
        score_mode: str = "label",
        mask_reduce: str | float = "mean",
        decision_threshold: float = 0.5,
        flip: bool = False,
        name: str | None = None,
    ) -> None:
        import torch
        from torchvision import transforms

        if score_mode not in _SCORE_MODES:
            raise ValueError(f"score_mode must be one of {sorted(_SCORE_MODES)}")

        self.device = resolve_device(device)
        self._torch = torch
        self._model = model.to(self.device).eval()
        self.image_size = int(image_size)
        self.score_mode = score_mode
        self.mask_reduce = mask_reduce
        self.decision_threshold = decision_threshold
        self.flip = flip
        if name:
            self.name = name

        # PSCC-Net TestData: RGB, /255, CHW — no normalization.
        self._transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), antialias=True),
            transforms.ToTensor(),
        ])

    # --- construction --------------------------------------------------

    @classmethod
    def from_module(cls, model: PSCCNet, **kwargs: Any) -> "PSCCNetDetector":
        """Wrap an already-built + already-loaded `PSCCNet`."""
        return cls(model, **kwargs)

    @classmethod
    def from_checkpoints(
        cls,
        hrnet_path: str | Path,
        nlc_path: str | Path,
        cls_path: str | Path,
        *,
        device: str = "auto",
        image_size: int = DEFAULT_INPUT_SIZE,
        score_mode: str = "label",
        mask_reduce: str | float = "mean",
        decision_threshold: float = 0.5,
        flip: bool = False,
    ) -> "PSCCNetDetector":
        """Build from the three individual PSCC-Net ``.pth`` files."""
        model = build_pscc_net(
            hrnet_path, nlc_path, cls_path,
            crop_size=(image_size, image_size),
        )
        return cls(
            model, device=device, image_size=image_size, score_mode=score_mode,
            mask_reduce=mask_reduce, decision_threshold=decision_threshold, flip=flip,
        )

    @classmethod
    def from_pretrained_dir(
        cls,
        directory: str | Path,
        **kwargs: Any,
    ) -> "PSCCNetDetector":
        """Build from a directory holding the three checkpoints (flat or the
        repo's ``<name>_checkpoint/`` nested layout)."""
        hrnet, nlc, cls_ = _find_checkpoints(Path(directory))
        return cls.from_checkpoints(hrnet, nlc, cls_, **kwargs)

    # keep the naming consistent with the other detectors in this repo
    from_checkpoint = from_pretrained_dir

    @classmethod
    def use_default(
        cls,
        *,
        device: str = "auto",
        weights_dir: str | Path | None = None,
        **kwargs: Any,
    ) -> "PSCCNetDetector":
        """Load PSCC-Net, downloading the three checkpoints from GitHub on first
        use (into ``src/weights/pscc_net/``)."""
        target = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        hrnet, nlc, cls_ = _ensure_default_weights(target)
        return cls.from_checkpoints(hrnet, nlc, cls_, device=device, **kwargs)

    # --- scoring -----------------------------------------------------

    def raw_output(self, image: Image.Image) -> dict[str, Any]:
        """PSCC-Net's full output for one image: ``mask`` (H x W tensor, [0, 1]),
        ``logit`` (2-vector), ``prob_forged`` (float)."""
        torch = self._torch
        x = self._transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self._model(x)
        return {
            "mask": out["mask"][0, 0].float().cpu(),
            "logit": out["logit"][0].float().cpu(),
            "prob_forged": float(out["prob_forged"][0].item()),
        }

    def predict_mask(self, image: Image.Image) -> Any:
        """The per-pixel manipulation mask as a NumPy ``float32`` array in
        ``[0, 1]``, resized back to the original image resolution. Handy for the
        fusion model and for visualizing *where* PSCC-Net thinks the edit is."""
        torch = self._torch
        mask = self.raw_output(image)["mask"].unsqueeze(0).unsqueeze(0)
        w, h = image.size
        mask = torch.nn.functional.interpolate(
            mask, size=(h, w), mode="bilinear", align_corners=True
        )
        return mask[0, 0].numpy()

    def _reduce_mask(self, mask: Any) -> float:
        torch = self._torch
        flat = mask.reshape(-1)
        if self.mask_reduce == "max":
            return float(flat.max().item())
        if isinstance(self.mask_reduce, (int, float)) and not isinstance(self.mask_reduce, bool):
            return float(torch.quantile(flat, float(self.mask_reduce)).item())
        return float(flat.mean().item())  # "mean" = fraction of the image flagged

    def _score(self, image: Image.Image) -> float:
        out = self.raw_output(image)
        p_label = out["prob_forged"]

        if self.score_mode == "label":
            p_ai = p_label
        elif self.score_mode == "mask":
            p_ai = self._reduce_mask(out["mask"])
        else:  # "hybrid"
            p_ai = max(p_label, float(out["mask"].reshape(-1).max().item()))

        p_ai = min(1.0, max(0.0, p_ai))
        return 1.0 - p_ai if self.flip else p_ai
