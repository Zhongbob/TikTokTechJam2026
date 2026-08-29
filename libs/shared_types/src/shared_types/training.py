"""Shared model-lifecycle contract for the training phase.

Pairs with `shared_types.interfaces` (AutoencoderRestorer / EnsembleDetector),
which is the "ready for inference" contract already wired into apps/web.
This module is the "not trained yet" counterpart: a small, framework-agnostic
base class each model owner extends in their Google Colab notebook.

Deliberately dependency-light (stdlib + PIL/numpy only, like the rest of
shared_types) so importing it doesn't drag torch/tensorflow/etc. into every
consumer — each concrete trainer pulls in whatever framework it needs inside
its own package's pyproject.toml. If you want a batteries-included training
loop instead of writing one by hand, see the note at the bottom of this file.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Iterable, TypeAlias, TypeVar

from PIL import Image

from shared_types.augmentation import AugmentationRecord, SourceMetadata


@dataclass
class LabeledImageSample:
    """One (image, label) training example for a classifier-style model.

    Shaped to match `data.datasets.load_sid_subset()`'s output 1:1 — see
    `data.datasets.to_labeled_samples()` to convert that function's
    (images, metadata) pair into a list of these.
    """

    image: Image.Image
    metadata: SourceMetadata


@dataclass
class ImagePairSample:
    """One (input, target) training pair for an image-restoration model.

    Shaped to match the manifest.json entries written by
    `data.datasets.AutoencoderDatasetBuilder.build()` — see
    `data.datasets.load_manifest_as_samples()` to load a built dataset
    directory straight into a list of these.
    """

    input_image: Image.Image
    target_image: Image.Image
    record: AugmentationRecord | None = None


@dataclass
class TrainingResult:
    """What `TrainableModel.train()` hands back once a run finishes."""

    epochs_completed: int = 0
    final_loss: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    checkpoint_path: str | None = None
    notes: str = ""


TSample = TypeVar("TSample")


class TrainableModel(ABC, Generic[TSample]):
    """Base class for a model in its "not trained yet" phase.

    Extend this in your model's package (packages/models/<name>/) and import
    the concrete class into your Colab notebook:

        trainer = MyModelTrainer()
        result = trainer.train(samples, epochs=20)
        trainer.save("my_model.ckpt")

    Internally, `train()` can do whatever it wants — call plain PyTorch,
    wrap a PyTorch Lightning `Trainer`, use a Hugging Face `Trainer`, etc.
    This base class only standardizes the *outer* shape so every model is
    driven the same way, regardless of what's inside.
    """

    name: str = "unnamed-trainable-model"

    @abstractmethod
    def train(self, samples: Iterable[TSample], **kwargs: Any) -> TrainingResult:
        """Run a full training loop over `samples` and report the result."""
        raise NotImplementedError

    def evaluate(self, samples: Iterable[TSample], **kwargs: Any) -> dict[str, float]:
        """Run the "testing" stage: score a trained model against held-out
        `samples` and return metric name -> value.

        Optional (not abstract) — override it if your model supports a
        distinct evaluation pass; the default just says it isn't wired up
        yet, so calling it on a model that hasn't implemented it fails loud
        instead of silently returning nothing.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement evaluate()")

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist trained weights/config to `path` for later inference use."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "TrainableModel[TSample]":
        """Reconstruct a trained instance from a checkpoint written by `save`."""
        raise NotImplementedError


# Convenience aliases so each model package can extend a name that already
# says what kind of data it trains on, instead of subscripting the generic
# directly: `class MyTrainer(ClassifierTrainableModel): ...`
ClassifierTrainableModel: TypeAlias = TrainableModel[LabeledImageSample]
AutoencoderTrainableModel: TypeAlias = TrainableModel[ImagePairSample]


# --- On an existing library ---------------------------------------------
# We deliberately did NOT build a full training loop (optimizer, batching,
# checkpointing schedule, ...) here — that's a solved problem, and different
# models will likely want different tools:
#   - PyTorch models: wrap a PyTorch Lightning `LightningModule` inside your
#     `train()` override and call `pl.Trainer(...).fit(...)` from it — you
#     still expose the same simple `.train(samples)` signature to Colab.
#   - Fine-tuning a pretrained classifier (ensemble / our_classifier): the
#     Hugging Face `transformers.Trainer` pairs naturally with the `datasets`
#     library already used by `data.datasets.load_sid_subset()`.
# Pick whichever fits inside your `train()` method; nothing above assumes
# either.
