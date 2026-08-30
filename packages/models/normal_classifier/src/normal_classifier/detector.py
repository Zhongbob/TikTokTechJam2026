"""The "running" stage: inference-ready wrapper around a trained
NormalClassifierTrainer checkpoint.

Implements `shared_types.interfaces.EnsembleDetector` — the same "ready"
contract apps/web already consumes (see
apps/web/src/web/services/factory.py's SWAP POINT comments), so this class
can be dropped straight in as a real detector once a checkpoint exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from shared_types import LabeledImageSample
from shared_types.detection import DetectionResult, EnsembleMemberResult
from shared_types.interfaces import EnsembleDetector

# Same mapping the trainer exports with: 0 = real, 1 = AI-generated/tampered.
_CLASS_LABELS = ("real", "ai_generated")


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
    """Extend/instantiate this once you have a trained checkpoint:

        from normal_classifier import NormalClassifierDetector

        detector = NormalClassifierDetector.from_checkpoint("normal_classifier.pt")
        result = detector.predict(some_pil_image)
        metrics = detector.evaluate(val_samples, generate_confusion_matrix=True)
    """

    name = "normal-classifier-yolo"
    is_placeholder = False

    def __init__(self, model: Any) -> None:
        self._model = model  # an ultralytics.YOLO instance

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "NormalClassifierDetector":
        from ultralytics import YOLO

        return cls(YOLO(str(path)))

    def predict(self, image: Image.Image, **kwargs: Any) -> DetectionResult:
        kwargs.setdefault("verbose", False)
        results = self._model.predict(image, **kwargs)
        probs = results[0].probs
        names: dict[int, str] = results[0].names
        name_to_index = {class_name: index for index, class_name in names.items()}
        if "ai_generated" not in name_to_index:
            raise ValueError(
                f"Loaded model's classes {list(names.values())} don't include 'ai_generated' — "
                "was it trained with NormalClassifierTrainer's class-folder layout?"
            )

        ai_generated_probability = float(probs.data[name_to_index["ai_generated"]])
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
