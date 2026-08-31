"""Inference wrapper around OpenSDI's MaskCLIP model (iamwangyabin/OpenSDI).

MaskCLIP = a frozen CLIP-ViT-L/14 visual encoder + a MAE-style side-adapter
network + a prompt-learned text head. ``forward(image, mask, label, edge_mask)``
returns a dict with:

  * ``pred_mask``  — pixel-level forgery probability map, sigmoid, image-sized;
  * ``pred_label`` — image-level class (``argmax`` of ``cls_features @ text``).

OpenSDI is really an image-manipulation-*localization* method (locally edited
real photos), so for a whole-image "is this AI-generated?" score this wrapper
reduces ``pred_mask`` to a scalar (fraction of the image flagged) by default,
or uses ``pred_label`` directly.

`torch` / `torchvision` are imported lazily.

⚠️ FIDELITY: MaskCLIP's class (``model/MaskCLIP.py``) plus ``prompt_learner`` /
``clip_utils`` are NOT vendored — they need the OpenSDI repo, plus
``IMDLBenCo`` and OpenAI ``clip``. Run ``opensdi_detector.setup_opensdi()`` (or
``python -m opensdi_detector.bootstrap``) once to clone the repo, install the
deps and download every weight; then ``OpenSDIDetector.use_default()`` just
works. For a manual setup, pass ``repo_dir=`` (a clone of iamwangyabin/OpenSDI)
or an ``arch_factory``.

``clip_utils.py`` also hard-``torch.load``s ``weights/mae_pretrain_vit_base.pth``
(relative to the cwd); the bootstrap places it there, and ``from_checkpoint()``
otherwise auto-downloads it (or copies from ``mae_weights=``).

The ``pred_label`` positive index and mask->probability reduction are
heuristics — validate on a labelled set and flip / switch ``score_mode`` if
needed.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any, Callable

from detector_common import ImageDetector, resolve_device
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_DIR = SCRIPT_DIR.parent / "weights"

# CLIP normalisation, matching OpenSDI test.py's post_transforms().
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_RESOLUTION = 512  # main_keys['ViTL']['resolution'] in model/MaskCLIP.py


#: OpenSDI's model/clip_utils.py hard-codes this relative path for the MAE
#: ViT-B pretrain weights (facebookresearch/mae).
_MAE_RELPATH = Path("weights/mae_pretrain_vit_base.pth")
_MAE_URL = "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"


def _ensure_mae_weights(mae_weights: str | os.PathLike[str] | None) -> None:
    """Guarantee ``./weights/mae_pretrain_vit_base.pth`` (relative to the cwd)
    exists before MaskCLIP() is built -- OpenSDI hard-codes that path. Copies
    from ``mae_weights`` if given, else downloads the ~330 MB checkpoint."""
    if _MAE_RELPATH.is_file():
        return
    _MAE_RELPATH.parent.mkdir(parents=True, exist_ok=True)
    if mae_weights is not None:
        source = Path(mae_weights).expanduser()
        if not source.is_file():
            raise FileNotFoundError(f"mae_weights={source} not found")
        import shutil

        shutil.copy(source, _MAE_RELPATH)
        return
    import torch

    print(f"[opensdi] downloading MAE ViT-B pretrain -> {_MAE_RELPATH.resolve()}")
    torch.hub.download_url_to_file(_MAE_URL, str(_MAE_RELPATH))


def _load_maskclip_class(repo_dir: str | os.PathLike[str] | None) -> Any:
    """Import ``MaskCLIP`` from an OpenSDI clone. ``repo_dir`` is appended to
    ``sys.path`` (not prepended) so it can't shadow this monorepo's own
    ``data`` package."""
    if repo_dir is not None:
        resolved = str(Path(repo_dir).resolve())
        if resolved not in sys.path:
            sys.path.append(resolved)
    try:
        import IMDLBenCo  # noqa: F401  (registry used by model/MaskCLIP.py)
        import clip  # noqa: F401  (OpenAI CLIP, not open_clip)
        module = importlib.import_module("model.MaskCLIP")
    except ModuleNotFoundError as error:
        raise ImportError(
            f"OpenSDI needs its repo and deps ({error}). Clone "
            "https://github.com/iamwangyabin/OpenSDI, `pip install IMDLBenCo "
            "git+https://github.com/openai/CLIP.git`, and pass repo_dir=<clone>."
        ) from error
    return module.MaskCLIP


class OpenSDIDetector(ImageDetector):
    """OpenSDI / MaskCLIP, wrapped as an `ImageDetector`.

        detector = OpenSDIDetector.from_checkpoint(
            "MaskCLIP_Si.pth", repo_dir="/content/OpenSDI"
        )
        result = detector.predict(pil_image)
        metrics = detector.evaluate(val_samples, generate_confusion_matrix=True)

    Args (scoring):
        score_mode: ``"mask"`` (default) reduces ``pred_mask`` to p(ai); ``"label"``
            uses the image-level ``pred_label`` head directly.
        mask_reduce: for ``score_mode="mask"`` — ``"max"`` (default: "is *any*
            region flagged?"), ``"mean"`` (fraction of the image flagged — needs
            a low ``decision_threshold``), or a float quantile in (0, 1).
        decision_threshold: p(ai) at/above which the verdict is "ai_generated"
            (default 0.5). ``mask_reduce="mean"`` typically needs ~0.05-0.2.
        positive_label_index: which ``pred_label`` value means "manipulated"
            (default 1).
        flip: invert the final score if it comes out reversed on labelled data.

    NOTE: OpenSDI localises *locally edited* regions (SD-inpainted real photos).
    On fully AI-generated images (nothing "edited" vs "original") it often
    produces a near-empty mask -> everything scored "real". Expect it to help
    on tamper-style fakes, not text-to-image ones; validate per SID class.
    """

    name = "opensdi-maskclip"
    is_placeholder = False

    def __init__(
        self,
        model: Any,
        *,
        device: str = "auto",
        resolution: int = _RESOLUTION,
        mean: tuple[float, float, float] = _CLIP_MEAN,
        std: tuple[float, float, float] = _CLIP_STD,
        score_mode: str = "mask",
        mask_reduce: str | float = "max",
        decision_threshold: float = 0.5,
        positive_label_index: int = 1,
        flip: bool = False,
        name: str | None = None,
    ) -> None:
        import torch
        from torchvision import transforms

        if score_mode not in {"mask", "label"}:
            raise ValueError("score_mode must be 'mask' or 'label'")
        self.device = resolve_device(device)
        self._torch = torch
        self._model = model.to(self.device).eval()
        self.decision_threshold = decision_threshold
        self.score_mode = score_mode
        self.mask_reduce = mask_reduce
        self.positive_label_index = positive_label_index
        self.flip = flip
        if name:
            self.name = name
        self._transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(list(mean), list(std)),
        ])

    # --- construction --------------------------------------------------

    @classmethod
    def from_module(cls, model: Any, **kwargs: Any) -> "OpenSDIDetector":
        """Wrap an already-built + already-loaded MaskCLIP ``nn.Module``."""
        return cls(model, **kwargs)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        repo_dir: str | os.PathLike[str] | None = None,
        model_setting_name: str = "ViTL",
        mae_weights: str | os.PathLike[str] | None = None,
        arch_factory: Callable[[], Any] | None = None,
        device: str = "auto",
        score_mode: str = "mask",
        mask_reduce: str | float = "max",
        decision_threshold: float = 0.5,
        positive_label_index: int = 1,
        flip: bool = False,
    ) -> "OpenSDIDetector":
        import torch

        if arch_factory is not None:
            model = arch_factory()
        else:
            maskclip_cls = _load_maskclip_class(repo_dir)
            _ensure_mae_weights(mae_weights)  # MaskCLIP() torch.load()s weights/mae_pretrain_vit_base.pth
            model = maskclip_cls(model_setting_name=model_setting_name)

        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        state = blob
        if isinstance(blob, dict):
            for key in ("model", "state_dict", "model_state_dict"):
                if isinstance(blob.get(key), dict):
                    state = blob[key]
                    break
        state = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state.items() if hasattr(v, "shape")
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        # the frozen CLIP backbone is loaded by clip.load() in __init__, so
        # missing 'clip.*' keys are expected.
        real_missing = [k for k in missing if not k.startswith("clip.")]
        print(
            f"[opensdi] loaded {Path(path).name}: {len(real_missing)} missing "
            f"(+{len(missing) - len(real_missing)} frozen clip.*), {len(unexpected)} unexpected"
        )
        if real_missing and len(real_missing) > 0.3 * max(1, len(state)):
            print("[opensdi] WARNING: many trainable weights did not load — check model_setting_name / checkpoint.")

        return cls(
            model, device=device, score_mode=score_mode, mask_reduce=mask_reduce,
            decision_threshold=decision_threshold, positive_label_index=positive_label_index, flip=flip,
        )

    @classmethod
    def use_default(
        cls,
        *,
        repo_dir: str | os.PathLike[str] | None = None,
        checkpoint_name: str = "maskclip_opensdi.pth",
        device: str = "auto",
        **score_kwargs: Any,
    ) -> "OpenSDIDetector":
        repo_dir = repo_dir or os.environ.get("OPENSDI_REPO")
        checkpoint = DEFAULT_WEIGHTS_DIR / checkpoint_name
        if not checkpoint.is_file():
            # fall back to any MaskCLIP*.pth the bootstrap script dropped in
            matches = sorted(DEFAULT_WEIGHTS_DIR.glob("MaskCLIP*.pth"))
            if matches:
                checkpoint = matches[-1]
        if not checkpoint.is_file() or not repo_dir:
            raise FileNotFoundError(
                "OpenSDI needs a checkpoint and the official repo. Run "
                "`python -m opensdi_detector.bootstrap` (or "
                "`from opensdi_detector import setup_opensdi; setup_opensdi()`), "
                "which clones https://github.com/iamwangyabin/OpenSDI and downloads "
                f"the MaskCLIP weights to {DEFAULT_WEIGHTS_DIR}. Otherwise set "
                "OPENSDI_REPO=<clone> and pass repo_dir=/checkpoint yourself."
            )
        return cls.from_checkpoint(checkpoint, repo_dir=repo_dir, device=device, **score_kwargs)

    # --- scoring -----------------------------------------------------

    def raw_output(self, image: Image.Image) -> dict[str, Any]:
        """MaskCLIP's full output dict for one image (``pred_mask`` / ``pred_label``
        / losses) — use it to work out the right ``score_mode`` / direction."""
        torch = self._torch
        x = self._transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        b, _, h, w = x.shape
        zeros = torch.zeros(b, 1, h, w, device=self.device)
        label = torch.zeros(b, dtype=torch.long, device=self.device)
        with torch.no_grad():
            return self._model(x, zeros, label, zeros)

    def _reduce_mask(self, mask_flat: Any) -> float:
        torch = self._torch
        if self.mask_reduce == "max":
            return float(mask_flat.max().item())
        if isinstance(self.mask_reduce, (int, float)) and not isinstance(self.mask_reduce, bool):
            return float(torch.quantile(mask_flat, float(self.mask_reduce)).item())
        return float(mask_flat.mean().item())

    def _score(self, image: Image.Image) -> float:
        output = self.raw_output(image)

        if self.score_mode == "label":
            pred = output["pred_label"].reshape(-1)[0].item()
            p_ai = 1.0 if int(pred) == self.positive_label_index else 0.0
        else:
            p_ai = self._reduce_mask(output["pred_mask"].float().reshape(-1))

        p_ai = min(1.0, max(0.0, p_ai))
        return 1.0 - p_ai if self.flip else p_ai
