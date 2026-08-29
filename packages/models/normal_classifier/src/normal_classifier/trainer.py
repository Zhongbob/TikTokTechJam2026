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
from typing import Any, Iterable

from shared_types import ClassifierTrainableModel, LabeledImageSample, TrainingResult

# SID-Set's binary_aigc_label (0 = real, 1 = AI-generated/tampered) maps
# straight onto the two class folders Ultralytics' classification trainer
# expects (`<data_root>/<split>/<class_name>/*.jpg`).
_CLASS_NAMES = {0: "real", 1: "ai_generated"}


def _label_folder_name(sample: LabeledImageSample) -> str:
    return _CLASS_NAMES[int(sample.metadata["binary_aigc_label"])]


def _export_to_class_folders(samples: Iterable[LabeledImageSample], split_dir: Path) -> int:
    """Writes samples out as `split_dir/<class_name>/NNNNNN.jpg`.

    Takes `samples` in a single pass and writes+discards each image as it
    goes — safe to call with a lazy generator (e.g. `iter_sid_subset()`)
    without ever holding the whole set in memory. Returns the total count
    written.
    """
    counts: dict[str, int] = {}
    for sample in samples:
        class_name = _label_folder_name(sample)
        folder = split_dir / class_name
        folder.mkdir(parents=True, exist_ok=True)
        index = counts.get(class_name, 0)
        sample.image.convert("RGB").save(folder / f"{index:06d}.jpg", quality=95)
        counts[class_name] = index + 1
    return sum(counts.values())


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
        from data.dataset_builder import iter_sid_subset, sid_subset_factory

        # Large, single-use training pull: iter_sid_subset() streams straight
        # through train() without ever holding the whole set in memory.
        train_samples = iter_sid_subset(images_per_label=1000, split="train")
        # Validation set is iterated more than once (train's val pass,
        # evaluate()); a factory re-streams it instead of materializing it.
        make_val = sid_subset_factory(images_per_label=200, split="validation")

        trainer = NormalClassifierTrainer()
        result = trainer.train(train_samples, val_samples=make_val(), epochs=100)
        trainer.save("normal_classifier.pt")

    Or just run this module for a small streamed smoke test:
        python -m normal_classifier.trainer --train-per-label 25 --epochs 5
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

        output_dir = Path(output_dir)
        if val_samples is None:
            # The stratified auto-split needs to see every sample's class
            # before it can decide where anything goes, so this path (only)
            # requires materializing `samples` into memory first. Pass
            # val_samples explicitly (e.g. from a separate iter_sid_subset()
            # call) to avoid this for a large, single-use training pull.
            samples = list(samples)
            if not samples:
                raise ValueError("samples must not be empty")
            samples, val_samples = _split_train_val(samples, val_fraction)

        # Single pass over each iterable, writing+discarding as we go — safe
        # to call with lazy generators (e.g. iter_sid_subset()) without ever
        # holding the full train/val sets in memory at once.
        train_count = _export_to_class_folders(samples, output_dir / "train")
        val_count = _export_to_class_folders(val_samples, output_dir / "val")
        if train_count == 0:
            raise ValueError("samples must not be empty")

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
            notes=f"Trained on {train_count} samples, validated on {val_count}.",
        )

    # --- testing ---------------------------------------------------------

    def evaluate(self, samples: Iterable[LabeledImageSample], **kwargs: Any) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("Call train() or load() before evaluate()")

        output_dir = Path(kwargs.pop("output_dir", "yolo_eval_dataset"))
        # Ultralytics' classification val() reads the same directory layout
        # as train(); a directory holding only a val/ split is sufficient.
        # Single pass, streaming-safe like train()'s export above.
        _export_to_class_folders(samples, output_dir / "val")
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


# --- CLI smoke run ------------------------------------------------------
# `python -m normal_classifier.trainer` (or running this file directly)
# streams a small balanced SID-Set subset straight from Hugging Face via the
# `data` library and does a short end-to-end train + evaluate, printing the
# results. Sizes and epochs are all flags so it stays quick by default.


def _build_cli_parser() -> "argparse.ArgumentParser":
    import argparse

    parser = argparse.ArgumentParser(
        description="Stream a small SID-Set subset and run a short train + evaluate of the baseline classifier.",
    )
    parser.add_argument("--train-per-label", type=int, default=25,
                        help="Training images streamed per SID label (real/synthetic/tampered).")
    parser.add_argument("--test-per-label", type=int, default=10,
                        help="Held-out images streamed per SID label for the evaluate() pass.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cpu", help="'cpu', '0', 'cuda:0', ...")
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--buffer-size", type=int, default=100,
                        help="iter_sid_subset() streaming-shuffle window.")
    parser.add_argument("--output-dir", default="sid_yolo_smoke")
    parser.add_argument("--base-weights", default="yolo26n-cls.pt")
    parser.add_argument("--hf-token", default=None)
    return parser


def _run_cli(argv: list[str] | None = None) -> None:
    from data.dataset_builder import iter_sid_subset, sid_subset_factory

    args = _build_cli_parser().parse_args(argv)

    print(
        f"Streaming SID-Set: {args.train_per_label}/label train, "
        f"{args.test_per_label}/label test | epochs={args.epochs} batch={args.batch} device={args.device}"
    )

    # Training pull: a single-use stream, consumed once by train() as it
    # writes each image to disk -- never more than one decoded image in RAM.
    train_samples = iter_sid_subset(
        args.train_per_label, seed=args.seed, buffer_size=args.buffer_size,
        split="train", hf_token=args.hf_token,
    )
    # Test pull: iterated by evaluate() -- a factory hands back a fresh stream
    # each call instead of materialising the set.
    make_test_samples = sid_subset_factory(
        args.test_per_label, seed=args.seed, buffer_size=args.buffer_size,
        split="validation", hf_token=args.hf_token,
    )

    trainer = NormalClassifierTrainer(base_weights=args.base_weights, image_size=args.image_size)
    result = trainer.train(
        train_samples,
        val_samples=make_test_samples(),
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
    )

    print("\n=== Training result ===")
    print(f"epochs completed : {result.epochs_completed}")
    print(f"checkpoint       : {result.checkpoint_path}")
    print(f"notes            : {result.notes}")
    if result.metrics:
        print("metrics:")
        for key, value in result.metrics.items():
            print(f"  {key}: {value:.4f}")

    metrics = trainer.evaluate(make_test_samples(), output_dir=f"{args.output_dir}_eval")
    print("\n=== Held-out evaluation ===")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    _run_cli()
