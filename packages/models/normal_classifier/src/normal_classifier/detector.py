"""The "running" stage: inference-ready wrapper around a trained
NormalClassifierTrainer checkpoint.

Implements `shared_types.interfaces.EnsembleDetector` — the same "ready"
contract apps/web already consumes (see
apps/web/src/web/services/factory.py's SWAP POINT comments), so this class
can be dropped straight in as a real detector once a checkpoint exists.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Iterable

from PIL import Image
from shared_types import LabeledImageSample
from shared_types.detection import DetectionResult, EnsembleMemberResult
from shared_types.interfaces import EnsembleDetector

# Same mapping the trainer exports with: 0 = real, 1 = AI-generated/tampered.
_CLASS_LABELS = ("real", "ai_generated")
SCRIPT_DIR = Path(__file__).resolve().parent

# Substrings that mark a class name as "AI-generated" / "real", used to figure
# out an arbitrary checkpoint's label convention when it isn't the trainer's.
_AI_HINTS = (
    "ai_generated", "ai-generated", "aigenerated", "aigc", "synthetic", "fake",
    "generated", "gan", "diffusion", "deepfake", "spoof", "tampered",
)
_REAL_HINTS = ("real", "authentic", "natural", "nature", "genuine", "pristine", "bonafide", "live")


def _canonical(name: str) -> str:
    """Lower-case a class name and drop a leading index prefix like ``0_`` / ``1-``."""
    return re.sub(r"^\s*[0-9]+\s*[_\-. ]*", "", str(name).strip().lower())


def _is_ai_label(canon: str) -> bool:
    return canon in ("ai", "1") or canon.startswith(("ai_", "ai-", "ai ")) or any(h in canon for h in _AI_HINTS)


def _is_real_label(canon: str) -> bool:
    return canon in ("real", "0") or any(h in canon for h in _REAL_HINTS)


def _as_index_map(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(v) for i, v in enumerate(names)}


def resolve_ai_scorer(
    names: Any, positive_class: int | str | None = None
) -> Callable[[Any], float]:
    """Return ``probs_vector -> p(ai_generated)`` for a model whose class list is
    `names` (a dict/list of class names).

    - ``positive_class`` given as an int: that index is the AI-generated class.
    - given as a str: matched exactly, then as a case-insensitive substring.
    - not given: inferred from the class names. If a single "real" class is
      found, ``p(ai) = 1 - p(real)`` (works for 2- and N-class heads); else the
      single AI-ish class is used. Raises if it can't tell — pass
      ``positive_class`` in that case.
    """
    index_by_name = _as_index_map(names)
    name_to_index = {v: k for k, v in index_by_name.items()}

    if isinstance(positive_class, int):
        pos = positive_class
        return lambda data: float(data[pos])
    if isinstance(positive_class, str):
        if positive_class in name_to_index:
            pos = name_to_index[positive_class]
        else:
            matches = [i for n, i in name_to_index.items() if positive_class.lower() in n.lower()]
            if len(matches) != 1:
                raise ValueError(
                    f"positive_class={positive_class!r} matched {len(matches)} of "
                    f"{list(index_by_name.values())}; pass an exact name or an int index."
                )
            pos = matches[0]
        return lambda data: float(data[pos])

    canon = {i: _canonical(n) for i, n in index_by_name.items()}
    real = [i for i, c in canon.items() if _is_real_label(c)]
    ai = [i for i, c in canon.items() if _is_ai_label(c)]

    if len(real) == 1:
        r = real[0]
        return lambda data: float(1.0 - data[r])
    if len(ai) == 1:
        a = ai[0]
        return lambda data: float(data[a])
    if len(index_by_name) == 2 and len(real) == 0 and len(ai) == 0:
        # last resort: assume the convention 0 = negative, 1 = positive
        import warnings

        pos = max(index_by_name)
        warnings.warn(
            f"Guessing class {pos} ({index_by_name[pos]!r}) is 'AI-generated' for {list(index_by_name.values())}. "
            "Pass positive_class= to be sure.",
            stacklevel=2,
        )
        return lambda data: float(data[pos])
    raise ValueError(
        f"Couldn't infer which of {list(index_by_name.values())} is the AI-generated class. "
        "Pass positive_class=<class name or int index> to NormalClassifierDetector "
        "(e.g. positive_class='1_synthetic' or positive_class=1)."
    )



def _save_confusion_matrix(confusion: list[list[int]], output_dir: Path, title: str) -> Path:
    """Write a 2x2 (real / ai_generated) confusion matrix under `output_dir`.

    A PNG via matplotlib when it is importable (it ships with ultralytics),
    otherwise a plain-text CSV table so this never hard-fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        path = output_dir / "confusion_matrix.txt"
        rows = [",".join(["true\\pred", *_CLASS_LABELS])]
        rows += [",".join([_CLASS_LABELS[i], *(str(v) for v in row)]) for i, row in enumerate(confusion)]
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path

    path = output_dir / "confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(confusion, cmap="Blues")
    ax.set_xticks(range(2))
    ax.set_xticklabels(_CLASS_LABELS)
    ax.set_yticks(range(2))
    ax.set_yticklabels(_CLASS_LABELS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    vmax = max((max(row) for row in confusion), default=1) or 1
    for i in range(2):
        for j in range(2):
            value = confusion[i][j]
            ax.text(j, i, str(value), ha="center", va="center",
                    color="white" if value > vmax / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


class NormalClassifierDetector(EnsembleDetector):
    """Wrap any Ultralytics YOLO *classification* checkpoint as a detector.

        from normal_classifier import NormalClassifierDetector

        # trained by NormalClassifierTrainer (classes: real / ai_generated)
        detector = NormalClassifierDetector.from_checkpoint("normal_classifier.pt")

        # someone else's checkpoint with different class names
        detector = NormalClassifierDetector.from_checkpoint(
            "their_model.pt", positive_class="1_synthetic"   # or positive_class=1
        )
        result = detector.predict(some_pil_image)
        metrics = detector.evaluate(val_samples, generate_confusion_matrix=True)

    ``positive_class`` says which class means "AI-generated": an exact class
    name, a case-insensitive substring, or an int index. Left as ``None`` the
    class is inferred from the names (see :func:`resolve_ai_scorer`).
    """

    name = "normal-classifier-yolo"
    is_placeholder = False

    #: Checkpoint shipped in this package, loaded by :meth:`use_default`.
    DEFAULT_CHECKPOINT = SCRIPT_DIR.parent / "weights" / "normal_classifier_augmented.pt"

    def __init__(
        self,
        model: Any,
        *,
        positive_class: int | str | None = None,
        name: str | None = None,
    ) -> None:
        self._model = model  # an ultralytics.YOLO instance
        self._positive_class = positive_class
        if name:
            self.name = name
        model_names = getattr(model, "names", None)
        # Resolve eagerly (fail fast on an ambiguous class list); fall back to
        # resolving on the first predict() if the model has no `.names` yet.
        self._scorer: Callable[[Any], float] | None = (
            resolve_ai_scorer(model_names, positive_class) if model_names else None
        )

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        positive_class: int | str | None = None,
        name: str | None = None,
    ) -> "NormalClassifierDetector":
        from ultralytics import YOLO

        return cls(YOLO(str(path)), positive_class=positive_class, name=name)

    @classmethod
    def use_default(cls, *, positive_class: int | str | None = None) -> "NormalClassifierDetector":
        """Load the checkpoint bundled with this package
        (``src/weights/normal_classifier_augmented.pt``)."""
        checkpoint = cls.DEFAULT_CHECKPOINT
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Default checkpoint not found at {checkpoint}. Train one with "
                "NormalClassifierTrainer and save it there, or call from_checkpoint(path)."
            )
        return cls.from_checkpoint(checkpoint, positive_class=positive_class)

    def predict(self, image: Image.Image, **kwargs: Any) -> DetectionResult:
        kwargs.setdefault("verbose", False)
        results = self._model.predict(image, **kwargs)
        probs = results[0].probs
        if self._scorer is None:
            self._scorer = resolve_ai_scorer(results[0].names, self._positive_class)

        ai_generated_probability = min(1.0, max(0.0, self._scorer(probs.data)))
        member = EnsembleMemberResult(
            model_name=self.name,
            ai_generated_probability=ai_generated_probability,
            confidence=float(probs.top1conf),
            is_placeholder=False,
        )
        return DetectionResult(
            verdict="ai_generated" if ai_generated_probability >= 0.5 else "real",
            ai_generated_probability=ai_generated_probability,
            member_results=(member,),
            is_placeholder=False,
            model_version=self.name,
        )

    def evaluate(
        self,
        samples: Iterable[LabeledImageSample],
        generate_confusion_matrix: bool = False,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Score the detector on labelled `samples` by running `predict()` on
        each image and comparing its verdict to the ground-truth
        `binary_aigc_label` (0 = real, 1 = ai_generated).

        Mirrors `NormalClassifierTrainer.evaluate()`'s calling convention:
        `samples` plus keyword extras, with `output_dir` (default
        "detector_eval") popped for artefacts; any other keyword args are
        forwarded to the underlying model's `predict()` (e.g. `device=`).

        When `generate_confusion_matrix` is True, a 2x2 confusion-matrix image
        is written under `output_dir` (PNG via matplotlib, or a `.txt` table
        if matplotlib is unavailable).

        Returns accuracy / precision / recall / f1 for the "ai_generated"
        class, plus the raw `tn` / `fp` / `fn` / `tp` counts and `n_samples`.
        """
        output_dir = Path(kwargs.pop("output_dir", "detector_eval"))

        iterator: Iterable[LabeledImageSample] = samples
        try:  # a determinate tqdm bar when the sample count is known
            from tqdm.auto import tqdm

            iterator = tqdm(samples, total=len(samples), unit="img", desc="evaluate")  # type: ignore[arg-type]
        except (ImportError, TypeError):
            pass

        # confusion[true][pred], 0 = real, 1 = ai_generated
        confusion = [[0, 0], [0, 0]]
        for sample in iterator:
            true_label = int(sample.metadata["binary_aigc_label"])
            predicted = 1 if self.predict(sample.image, **kwargs).verdict == "ai_generated" else 0
            confusion[true_label][predicted] += 1

        total = sum(sum(row) for row in confusion)
        if total == 0:
            raise ValueError("samples must not be empty")

        tn, fp = confusion[0]
        fn, tp = confusion[1]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        metrics = {
            "accuracy": (tp + tn) / total,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
            "tp": float(tp),
            "n_samples": float(total),
        }

        if generate_confusion_matrix:
            path = _save_confusion_matrix(confusion, output_dir, f"{self.name} confusion matrix")
            print(f"Confusion matrix written to {path}")

        return metrics

    