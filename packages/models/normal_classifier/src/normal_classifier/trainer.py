"""Training + testing stages for the baseline real-vs-AI-generated classifier.

Wraps Ultralytics YOLO classification (fine-tuning yolo26n-cls.pt), the same
model used in notebooks/baseline.ipynb — this class is the direct,
class-based replacement for that notebook's inline code.

`ultralytics` (and the torch/torchvision it pulls in) is imported lazily
inside the methods that need it, so importing this module — or building
tooling against it — doesn't require those heavy deps to be installed.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Iterable, Sequence

from shared_types import ClassifierTrainableModel, LabeledImageSample, TrainingResult

# SID-Set's binary_aigc_label (0 = real, 1 = AI-generated/tampered) maps
# straight onto the two class folders Ultralytics' classification trainer
# expects (`<data_root>/<split>/<class_name>/*.jpg`).
_CLASS_NAMES = {0: "real", 1: "ai_generated"}


def _label_folder_name(sample: LabeledImageSample) -> str:
    return _CLASS_NAMES[int(sample.metadata["binary_aigc_label"])]


def _export_to_class_folders(samples: Sequence[LabeledImageSample], split_dir: Path) -> None:
    """Writes samples out as `split_dir/<class_name>/NNNNNN.jpg`."""
    counts: dict[str, int] = {}
    for sample in samples:
        class_name = _label_folder_name(sample)
        folder = split_dir / class_name
        folder.mkdir(parents=True, exist_ok=True)
        index = counts.get(class_name, 0)
        sample.image.convert("RGB").save(folder / f"{index:06d}.jpg", quality=95)
        counts[class_name] = index + 1


def _split_train_val(
    samples: list[LabeledImageSample], val_fraction: float
) -> tuple[list[LabeledImageSample], list[LabeledImageSample]]:
    """Splits off `val_fraction` of samples *per class*, not just the tail of
    the list — otherwise a caller passing already-grouped-by-class samples
    (e.g. all "real" first) would get a validation set of a single class.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")

    by_class: dict[str, list[LabeledImageSample]] = {}
    for sample in samples:
        by_class.setdefault(_label_folder_name(sample), []).append(sample)

    train_samples: list[LabeledImageSample] = []
    val_samples: list[LabeledImageSample] = []
    for class_samples in by_class.values():
        split_index = max(1, int(len(class_samples) * (1 - val_fraction))) if len(class_samples) > 1 else 1
        train_samples.extend(class_samples[:split_index])
        val_samples.extend(class_samples[split_index:])
    return train_samples, val_samples


class NormalClassifierTrainer(ClassifierTrainableModel):
    """Extend/instantiate this in your Colab notebook:

        from normal_classifier import NormalClassifierTrainer
        from data.datasets import load_sid_subset, to_labeled_samples

        train_images, train_meta = load_sid_subset(images_per_label=1000, split="train")
        val_images, val_meta = load_sid_subset(images_per_label=200, split="validation")

        trainer = NormalClassifierTrainer()
        result = trainer.train(
            to_labeled_samples(train_images, train_meta),
            val_samples=to_labeled_samples(val_images, val_meta),
            epochs=100,
        )
        trainer.save("normal_classifier.pt")
    """

    name = "normal-classifier-yolo"

    def __init__(self, base_weights: str = "yolo26n-cls.pt", image_size: int = 224) -> None:
        self.base_weights = base_weights
        self.image_size = image_size
        self._model = None  # ultralytics.YOLO, created on first train()/load()

    # --- training (Colab) ----------------------------------------------

    def train(
        self,
        samples: Iterable[LabeledImageSample],
        *,
        val_samples: Iterable[LabeledImageSample] | None = None,
        val_fraction: float = 0.2,
        output_dir: str | Path = "yolo_dataset",
        epochs: int = 100,
        batch: int = 32,
        patience: int = 10,
        device: str = "cpu",
        plots: bool = True,
        **kwargs: Any,
    ) -> TrainingResult:
        from ultralytics import YOLO

        samples = list(samples)
        if not samples:
            raise ValueError("samples must not be empty")
        if val_samples is None:
            samples, val_samples = _split_train_val(samples, val_fraction)
        else:
            val_samples = list(val_samples)

        output_dir = Path(output_dir)
        _export_to_class_folders(samples, output_dir / "train")
        _export_to_class_folders(val_samples, output_dir / "val")

        self._model = YOLO(self.base_weights)
        results = self._model.train(
            data=str(output_dir),
            epochs=epochs,
            imgsz=self.image_size,
            batch=batch,
            patience=patience,
            device=device,
            plots=plots,
            **kwargs,
        )

        # Best-effort metric/checkpoint extraction — Ultralytics' exact
        # result-object attributes have drifted across versions, so this
        # degrades gracefully rather than raising if one is missing.
        results_dict = getattr(results, "results_dict", None) or {}
        trainer = getattr(self._model, "trainer", None)
        checkpoint = getattr(trainer, "best", None)
        epochs_completed = getattr(trainer, "epoch", None)

        return TrainingResult(
            epochs_completed=int(epochs_completed) + 1 if epochs_completed is not None else epochs,
            metrics={k: float(v) for k, v in results_dict.items() if isinstance(v, (int, float))},
            checkpoint_path=str(checkpoint) if checkpoint else None,
            notes=f"Trained on {len(samples)} samples, validated on {len(val_samples)}.",
        )

    # --- testing ---------------------------------------------------------

    def evaluate(self, samples: Iterable[LabeledImageSample], **kwargs: Any) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("Call train() or load() before evaluate()")

        output_dir = Path(kwargs.pop("output_dir", "yolo_eval_dataset"))
        # Ultralytics' classification val() reads the same directory layout
        # as train(); a directory holding only a val/ split is sufficient.
        _export_to_class_folders(list(samples), output_dir / "val")
        metrics = self._model.val(data=str(output_dir), **kwargs)

        results_dict = getattr(metrics, "results_dict", None) or {}
        if results_dict:
            return {k: float(v) for k, v in results_dict.items() if isinstance(v, (int, float))}
        # Fall back to the classification-specific top1/top5 accuracy fields.
        return {
            name: float(value)
            for name, value in (("top1", getattr(metrics, "top1", None)), ("top5", getattr(metrics, "top5", None)))
            if value is not None
        }

    # --- persistence -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("Nothing trained yet — call train() first")
        checkpoint = getattr(getattr(self._model, "trainer", None), "best", None)
        if checkpoint is None:
            raise RuntimeError("No checkpoint found on the trained model — did train() finish?")
        shutil.copy(checkpoint, path)

    @classmethod
    def load(cls, path: str | Path) -> "NormalClassifierTrainer":
        from ultralytics import YOLO

        instance = cls(base_weights=str(path))
        instance._model = YOLO(str(path))
        return instance
