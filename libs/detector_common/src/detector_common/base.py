"""Base class + helpers shared by every single-model image detector."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from shared_types import LabeledImageSample
from shared_types.detection import DetectionResult, EnsembleMemberResult

_CLASS_LABELS = ("real", "ai_generated")


def resolve_device(device: str = "auto") -> str:
    """Turn ``"auto"`` into ``"cuda"`` / ``"mps"`` / ``"cpu"``; pass anything
    else straight through. Imports torch lazily so this module stays light."""
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_confusion_matrix(confusion: list[list[int]], output_dir: Path, title: str) -> Path:
    """Write a 2x2 (real / ai_generated) confusion matrix under ``output_dir``.

    A PNG via matplotlib when it is importable, otherwise a plain-text CSV
    table so this never hard-fails.
    """
    output_dir = Path(output_dir)
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


class ImageDetector(ABC):
    """Common surface for a single AI-generated-image detector.

    Subclasses set ``name`` and implement ``_score()``. Everything the
    ``EnsembleDetector`` protocol and the notebooks expect (``predict`` /
    ``evaluate``) is derived from that.
    """

    #: Stable identifier, also used as the ``model_version`` / eval output dir.
    name: str = "image-detector"
    #: False for a real trained model, True for a stand-in.
    is_placeholder: bool = False
    #: p(ai_generated) at or above this counts as "ai_generated".
    decision_threshold: float = 0.5

    # --- to implement in subclasses ------------------------------------

    @abstractmethod
    def _score(self, image: Image.Image) -> float:
        """Return p(ai_generated) in ``[0, 1]`` for one RGB image."""

    # --- derived interface -------------------------------------------

    @staticmethod
    def _confidence(p_ai: float) -> float:
        """Self-reported certainty: distance from the 0.5 fence, scaled to 0..1."""
        return min(1.0, abs(p_ai - 0.5) * 2.0)

    def predict(self, image: Image.Image) -> DetectionResult:
        p_ai = float(self._score(image.convert("RGB")))
        p_ai = min(1.0, max(0.0, p_ai))
        member = EnsembleMemberResult(
            model_name=self.name,
            ai_generated_probability=p_ai,
            confidence=self._confidence(p_ai),
            is_placeholder=self.is_placeholder,
        )
        return DetectionResult(
            verdict="ai_generated" if p_ai >= self.decision_threshold else "real",
            ai_generated_probability=p_ai,
            member_results=(member,),
            is_placeholder=self.is_placeholder,
            model_version=self.name,
        )

    def evaluate(
        self,
        samples: Iterable[LabeledImageSample],
        generate_confusion_matrix: bool = False,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Score the detector on labelled `samples` by running `predict()` on
        each and comparing the verdict to the ground-truth `binary_aigc_label`
        (0 = real, 1 = ai_generated).

        Same calling convention as `NormalClassifierDetector.evaluate()`:
        `samples` plus keyword extras, with `output_dir` (default
        ``"<name>_eval"``) popped for artefacts.

        When `generate_confusion_matrix` is True, a 2x2 confusion-matrix image
        is written under `output_dir`.

        Returns accuracy / precision / recall / f1 for the "ai_generated"
        class, plus raw `tn` / `fp` / `fn` / `tp` counts and `n_samples`.
        """
        output_dir = Path(kwargs.pop("output_dir", f"{self.name}_eval"))

        iterator: Iterable[LabeledImageSample] = samples
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(samples, total=len(samples), unit="img", desc=f"evaluate {self.name}")  # type: ignore[arg-type]
        except (ImportError, TypeError):
            pass

        confusion = [[0, 0], [0, 0]]  # confusion[true][pred]
        for sample in iterator:
            true_label = int(sample.metadata["binary_aigc_label"])
            predicted = 1 if self.predict(sample.image).verdict == "ai_generated" else 0
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
            path = save_confusion_matrix(confusion, output_dir, f"{self.name} confusion matrix")
            print(f"Confusion matrix written to {path}")

        return metrics
