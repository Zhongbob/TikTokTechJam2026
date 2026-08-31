"""Base class + helpers shared by every single-model image detector."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from shared_types import LabeledImageSample
from shared_types.detection import DetectionResult, EnsembleMemberResult

_CLASS_LABELS = ("real", "ai_generated")


def _roc_auc(positives: list[float], negatives: list[float]) -> float:
    """Threshold-free separability (Mann-Whitney U). 1.0 = perfect, 0.5 = chance,
    < 0.5 = the score is inverted. Pure-python, no sklearn."""
    if not positives or not negatives:
        return float("nan")
    combined = sorted([(s, 1) for s in positives] + [(s, 0) for s in negatives])
    ranks: dict[int, float] = {}
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        average = (i + j - 1) / 2 + 1
        for k in range(i, j):
            ranks[k] = average
        i = j
    rank_sum_pos = sum(ranks[idx] for idx, (_, label) in enumerate(combined) if label == 1)
    n_pos, n_neg = len(positives), len(negatives)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


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
        *,
        decision_threshold: float | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Score the detector on labelled `samples` by running `predict()` on
        each and comparing ``ai_generated_probability`` to a threshold against
        the ground-truth `binary_aigc_label` (0 = real, 1 = ai_generated).

        `decision_threshold` overrides ``self.decision_threshold`` for this call
        only. `output_dir` (default ``"<name>_eval"``) is popped for artefacts.

        Returns accuracy / precision / recall / f1 / **roc_auc** (threshold-free
        separability — 1.0 perfect, ~0.5 no signal, <0.5 inverted) at the chosen
        threshold, plus `mean_score_real` / `mean_score_ai`, the raw
        `tn` / `fp` / `fn` / `tp` counts and `n_samples`. Prints a warning when
        the two classes' scores barely differ or the AUC is < 0.5.
        """
        output_dir = Path(kwargs.pop("output_dir", f"{self.name}_eval"))
        threshold = self.decision_threshold if decision_threshold is None else float(decision_threshold)

        iterator: Iterable[LabeledImageSample] = samples
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(samples, total=len(samples), unit="img", desc=f"evaluate {self.name}")  # type: ignore[arg-type]
        except (ImportError, TypeError):
            pass

        scores: list[float] = []
        trues: list[int] = []
        for sample in iterator:
            trues.append(int(sample.metadata["binary_aigc_label"]))
            scores.append(float(self.predict(sample.image).ai_generated_probability))

        total = len(trues)
        if total == 0:
            raise ValueError("samples must not be empty")

        confusion = [[0, 0], [0, 0]]  # confusion[true][pred]
        for true_label, score in zip(trues, scores):
            confusion[true_label][1 if score >= threshold else 0] += 1

        tn, fp = confusion[0]
        fn, tp = confusion[1]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        ai_scores = [s for s, t in zip(scores, trues) if t == 1]
        real_scores = [s for s, t in zip(scores, trues) if t == 0]
        mean_ai = sum(ai_scores) / len(ai_scores) if ai_scores else float("nan")
        mean_real = sum(real_scores) / len(real_scores) if real_scores else float("nan")
        auc = _roc_auc(ai_scores, real_scores)

        metrics = {
            "accuracy": (tp + tn) / total,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "roc_auc": auc,
            "threshold": threshold,
            "mean_score_real": mean_real,
            "mean_score_ai": mean_ai,
            "tn": float(tn),
            "fp": float(fp),
            "fn": float(fn),
            "tp": float(tp),
            "n_samples": float(total),
        }

        if ai_scores and real_scores:
            if abs(mean_ai - mean_real) < 0.02:
                print(
                    f"[{self.name}] WARNING: p(ai) barely differs between classes "
                    f"(real {mean_real:.3f} vs ai {mean_ai:.3f}) — the model is not discriminating on this data."
                )
            elif auc < 0.5:
                print(
                    f"[{self.name}] NOTE: roc_auc {auc:.3f} < 0.5 — scores look inverted; "
                    "construct the detector with flip=True (or swap the positive index)."
                )

        if generate_confusion_matrix:
            path = save_confusion_matrix(confusion, output_dir, f"{self.name} confusion matrix")
            print(f"Confusion matrix written to {path}")

        return metrics
