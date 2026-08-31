"""Inference wrapper for the in-house fine-tuned CLIP ViT-B/32 AIGC classifier.

The checkpoint (``clip_vit_b32_best*.pt`` at the repo root) fine-tunes the last
few visual transformer blocks + ``visual.ln_post`` + ``visual.proj`` +
``logit_scale`` of OpenAI CLIP ViT-B/32. It classifies **by CLIP text-image
similarity** against per-class prompt sets (``class_prompts`` in the checkpoint),
not a linear head::

    img_feat  = normalise(model.encode_image(preprocess(image)))
    text_feat = normalise(mean_p normalise(model.encode_text(prompts[class])))   # per class
    logits    = logit_scale.exp() * img_feat @ text_feat.T                       # [1, n_classes]
    p(class)  = softmax(logits)

Checkpoint metadata: ``base_model_name`` ("ViT-B/32"), ``class_names``
({0: "real", 1: "AI-generated/tampered"}), ``class_prompts``, ``trained_state_dict``.

Needs OpenAI CLIP: ``pip install ftfy regex git+https://github.com/openai/CLIP.git``.
`torch` / `clip` are imported lazily.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from detector_common import ImageDetector, resolve_device
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
#: bundled checkpoint dir (populated if the .pt is < 100 MiB) then repo root
DEFAULT_WEIGHTS_DIR = SCRIPT_DIR.parent / "weights"
#: packages/models/clip_vit_b32/src/clip_vit_b32 -> repo root
REPO_ROOT = SCRIPT_DIR.parents[4]

DEFAULT_BASE_MODEL = "ViT-B/32"
_DEFAULT_CLASS_NAMES = {0: "real", 1: "synthetic"}
_AI_HINTS = ("synthetic", "fake", "ai", "generated", "gan", "diffusion", "tampered", "aigc", "deepfake")
_REAL_HINTS = ("real", "authentic", "natural", "genuine", "pristine", "photograph")


def _int_keyed(names: Any) -> dict[int, str]:
    if not isinstance(names, dict):
        return {i: str(n) for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def _resolve_positive_index(class_names: dict[int, str], override: int | str | None) -> int:
    if isinstance(override, int):
        return override
    if isinstance(override, str):
        for index, name in class_names.items():
            if override.lower() in name.lower():
                return index
        raise ValueError(f"positive_class={override!r} not in {list(class_names.values())}")
    real = [i for i, n in class_names.items() if any(h in n.lower() for h in _REAL_HINTS)]
    if len(real) == 1 and len(class_names) == 2:
        return next(i for i in class_names if i != real[0])
    ai = [i for i, n in class_names.items() if any(h in n.lower() for h in _AI_HINTS)]
    if len(ai) == 1:
        return ai[0]
    return max(class_names)


class ClipViTB32Detector(ImageDetector):
    """Fine-tuned CLIP ViT-B/32 (prompt-similarity), wrapped as an `ImageDetector`.

    ``positive_class`` overrides which class means "AI-generated" (int index or a
    class-name substring); by default inferred from the checkpoint's
    ``class_names``. ``flip=True`` inverts the final score if a labelled eval
    comes out reversed.
    """

    name = "clip-vit-b32-aigc"
    is_placeholder = False

    def __init__(
        self,
        model: Any,
        preprocess: Any,
        text_features: Any,
        *,
        class_names: dict[int, str],
        device: str = "auto",
        positive_index: int = 1,
        flip: bool = False,
    ) -> None:
        import torch

        self.device = resolve_device(device)
        self._torch = torch
        self._model = model.to(self.device).eval()
        self._preprocess = preprocess
        self._text_features = text_features.to(self.device)  # [n_classes, embed_dim], L2-normalised
        self.class_names = class_names
        self._positive_index = positive_index
        self.flip = flip

    # --- construction --------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "auto",
        positive_class: int | str | None = None,
        flip: bool = False,
    ) -> "ClipViTB32Detector":
        import torch

        try:
            import clip
        except ImportError as error:  # pragma: no cover
            raise ImportError(
                "clip_vit_b32 needs OpenAI CLIP: "
                "pip install ftfy regex git+https://github.com/openai/CLIP.git"
            ) from error

        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        _meta = blob if isinstance(blob, dict) else {}
        base_model = _meta.get("base_model_name", DEFAULT_BASE_MODEL)
        model, preprocess = clip.load(base_model, device="cpu")
        model.float()

        state = _meta.get("trained_state_dict", blob)
        state = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state.items()
            if hasattr(v, "shape")
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        loaded = len(model.state_dict()) - len(missing)
        print(f"[clip-vit-b32] loaded {loaded} fine-tuned tensors from {Path(path).name} "
              f"({len(unexpected)} unexpected)")
        if loaded < len(state):
            print(f"[clip-vit-b32] WARNING: {len(state) - loaded} checkpoint tensors did not match the CLIP arch")

        class_names = _int_keyed(_meta.get("class_names") or _DEFAULT_CLASS_NAMES)
        prompts_raw = _meta.get("class_prompts") or {i: [name] for i, name in class_names.items()}
        prompts = {int(k): list(v) for k, v in prompts_raw.items()}

        model.eval()
        per_class: list[Any] = []
        with torch.no_grad():
            for index in sorted(class_names):
                tokens = clip.tokenize(prompts[index])
                feats = model.encode_text(tokens).float()
                feats = feats / feats.norm(dim=-1, keepdim=True)
                mean = feats.mean(dim=0)
                per_class.append(mean / mean.norm())
        text_features = torch.stack(per_class, dim=0)  # [n_classes, embed_dim]

        positive_index = _resolve_positive_index(class_names, positive_class)
        return cls(
            model, preprocess, text_features,
            class_names=class_names, device=device, positive_index=positive_index, flip=flip,
        )

    from_pretrained = from_checkpoint

    @classmethod
    def use_default(
        cls,
        *,
        device: str = "auto",
        checkpoint: str | Path | None = None,
        **kwargs: Any,
    ) -> "ClipViTB32Detector":
        """``checkpoint=`` takes an explicit ``.pt`` path; otherwise the first
        ``clip_vit_b32*.pt`` in ``src/weights/`` then the repo root."""
        if checkpoint is not None:
            return cls.from_checkpoint(Path(checkpoint).expanduser(), device=device, **kwargs)
        for directory in (DEFAULT_WEIGHTS_DIR, REPO_ROOT):
            for pattern in ("clip_vit_b32*.pt", "clip*vit*b32*.pt", "clip_vit_b32*.pth"):
                hits = sorted(directory.glob(pattern))
                if hits:
                    return cls.from_checkpoint(hits[0], device=device, **kwargs)
        raise FileNotFoundError(
            f"No CLIP ViT-B/32 checkpoint (clip_vit_b32*.pt) found in "
            f"{DEFAULT_WEIGHTS_DIR} or {REPO_ROOT}. Pass checkpoint=<path>, put it "
            "in one of them, or call from_checkpoint(path)."
        )

    # --- scoring -----------------------------------------------------

    def raw_output(self, image: Image.Image) -> Any:
        """CLIP similarity logits (scaled) over the classes for one image."""
        torch = self._torch
        tensor = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feats = self._model.encode_image(tensor).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            scale = self._model.logit_scale.exp()
            return (scale * feats @ self._text_features.t()).reshape(-1)

    def _score(self, image: Image.Image) -> float:
        torch = self._torch
        logits = self.raw_output(image)
        p_ai = float(torch.softmax(logits, dim=-1)[self._positive_index].item())
        return 1.0 - p_ai if self.flip else p_ai
