"""Inference wrapper for a trained augmentation-reversal autoencoder.

This class satisfies the `shared_types.interfaces.AutoencoderRestorer`
contract used by the app pipeline: it accepts an augmented image and returns
an image closer to the original clean version.

`restore_stream()` / `RestoredStream` take a stream of `LabeledImageSample`s
(e.g. `data.dataset_builder.augmented_sid_dataset(...)` /
`eval_dataset(..., augment=True)`), run every image back through the
autoencoder, and yield restored `LabeledImageSample`s with the metadata
untouched -- a drop-in `Iterable[LabeledImageSample]` for any detector's
`predict()` / `evaluate()` (the "transform reversal" pipeline).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch
from PIL import Image

from shared_types import ImagePairSample, LabeledImageSample
from shared_types.interfaces import AutoencoderRestorer

from autoencoder.trainer import SimpleImageAutoencoder

SCRIPT_DIR = Path(__file__).resolve().parent


class AutoencoderRestorerImpl(AutoencoderRestorer):
    """Load a trained restoration checkpoint and restore augmented images."""

    name = "autoencoder-augmentation-reversal"
    is_placeholder = False

    DEFAULT_CHECKPOINT = SCRIPT_DIR.parent / "weights" / "autoencoder_best.pt"

    def __init__(self, model: torch.nn.Module | None = None, image_size: int = 224, device: str = "cpu") -> None:
        self.image_size = image_size
        self.device = torch.device(device)
        self._model = model
        if self._model is not None:
            self._model.to(self.device)

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, image_size: int = 224, device: str = "cpu") -> "AutoencoderRestorerImpl":
        checkpoint = torch.load(path, map_location="cpu")
        model = SimpleImageAutoencoder(hidden_channels=int(checkpoint.get("hidden_channels", 32)))
        model.load_state_dict(checkpoint["model_state"])
        return cls(model=model, image_size=int(checkpoint.get("image_size", image_size)), device=device)

    @classmethod
    def use_default(cls) -> "AutoencoderRestorerImpl":
        checkpoint = cls.DEFAULT_CHECKPOINT
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Default checkpoint not found at {checkpoint}. Train one with AutoencoderTrainer and save it there or call from_checkpoint(path)."
            )
        return cls.from_checkpoint(checkpoint)

    def restore_batch(
        self, images: Sequence[Image.Image], *, keep_size: bool = True
    ) -> list[Image.Image]:
        """Restore a list of images in one forward pass.

        ``keep_size=True`` resizes each output back to its input's resolution;
        ``False`` leaves them at ``image_size`` (a detector will resize anyway,
        so this saves a round-trip).
        """
        if self._model is None:
            raise RuntimeError("Call use_default()/from_checkpoint() or pass a model before restoring")
        if not images:
            return []

        sizes = [im.size for im in images]
        arrays = [
            np.asarray(
                im.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC),
                dtype=np.float32,
            ) / 255.0
            for im in images
        ]
        batch = torch.from_numpy(np.stack(arrays).transpose(0, 3, 1, 2)).to(self.device)

        self._model.eval()
        with torch.no_grad():
            restored = self._model(batch).clamp(0.0, 1.0).cpu().permute(0, 2, 3, 1).numpy()

        outputs: list[Image.Image] = []
        for array, (width, height) in zip(restored, sizes):
            uint8 = np.clip(array * 255.0, 0, 255).round().astype(np.uint8)
            image = Image.fromarray(uint8, mode="RGB")
            if keep_size and (width, height) != (self.image_size, self.image_size):
                image = image.resize((width, height), Image.Resampling.BICUBIC)
            outputs.append(image)
        return outputs

    def predict(self, image: Image.Image, **kwargs: Any) -> Image.Image:
        return self.restore_batch([image], keep_size=True)[0]

    # `AutoencoderRestorer` protocol name.
    restore = predict

    def restore_stream(
        self,
        samples: Iterable[LabeledImageSample],
        *,
        batch_size: int = 16,
        keep_size: bool = True,
        progress: bool = False,
    ) -> Iterator[LabeledImageSample]:
        """Restore every sample's image, yielding `LabeledImageSample`s with the
        same metadata -- a memory-bounded, single-pass
        `Iterable[LabeledImageSample]` any detector accepts.

        Images are restored ``batch_size`` at a time. For a re-iterable version
        (so several detectors can consume the same restored set without
        re-running the autoencoder each time) use `restored_dataset()`.
        """
        iterator: Iterable[LabeledImageSample] = samples
        if progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(samples, desc=f"{self.name} restore", unit="img")  # type: ignore[arg-type]
            except ImportError:
                pass

        pending: list[LabeledImageSample] = []
        for sample in iterator:
            pending.append(sample)
            if len(pending) >= batch_size:
                yield from self._restore_pending(pending, keep_size)
                pending = []
        if pending:
            yield from self._restore_pending(pending, keep_size)

    def _restore_pending(
        self, pending: list[LabeledImageSample], keep_size: bool
    ) -> Iterator[LabeledImageSample]:
        restored = self.restore_batch([s.image for s in pending], keep_size=keep_size)
        for sample, image in zip(pending, restored):
            yield LabeledImageSample(image=image, metadata=sample.metadata)

    def restored_dataset(
        self,
        source: Iterable[LabeledImageSample],
        *,
        batch_size: int = 16,
        keep_size: bool = True,
        progress: bool = False,
    ) -> "RestoredStream":
        """A re-iterable view of ``source`` with every image run through this
        restorer (see `RestoredStream`)."""
        return RestoredStream(
            source, self, batch_size=batch_size, keep_size=keep_size, progress=progress
        )

    def evaluate(self, samples: Iterable[ImagePairSample], **kwargs: Any) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("Call load()/from_checkpoint() or initialize with a model before evaluate()")

        batch_size = int(kwargs.pop("batch_size", 16))
        criterion = torch.nn.MSELoss()
        from torch.utils.data import DataLoader

        from autoencoder.trainer import PairedImageDataset

        loader = DataLoader(PairedImageDataset(samples, image_size=self.image_size), batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        total_count = 0

        self._model.eval()
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device, dtype=torch.float32)
                targets = targets.to(self.device, dtype=torch.float32)
                outputs = self._model(inputs)
                total_loss += criterion(outputs, targets).item() * inputs.size(0)
                total_count += inputs.size(0)
        mse = total_loss / max(total_count, 1)
        return {"mse": float(mse), "rmse": float(np.sqrt(mse))}


# Backwards-compatible alias used by the app/service layer.
AutoencoderRestorer = AutoencoderRestorerImpl


class RestoredStream:
    """Re-iterable `Iterable[LabeledImageSample]` = a source stream with every
    image run back through an autoencoder restorer.

    ``source`` should itself be re-iterable (e.g. a
    `data.dataset_builder.StreamingAugmentedDataset` from
    `augmented_sid_dataset(...)` / `eval_dataset(..., augment=True)`); each
    ``iter()`` re-streams it and restores afresh. ``__len__`` and ``warm_cache``
    are forwarded to the source when it has them, so the usual
    "download/cache once, then iterate" flow still works::

        aug = eval_dataset(augment=True, num_augmentations=6, output_size=(224, 224),
                           cache_dir="wildfake_cache")
        restored = AutoencoderRestorer.use_default().restored_dataset(aug)
        restored.warm_cache()                 # caches the *augmented* source bytes
        detector.evaluate(restored, generate_confusion_matrix=True)
    """

    def __init__(
        self,
        source: Iterable[LabeledImageSample],
        restorer: "AutoencoderRestorerImpl",
        *,
        batch_size: int = 16,
        keep_size: bool = True,
        progress: bool = False,
    ) -> None:
        self._source = source
        self._restorer = restorer
        self._batch_size = batch_size
        self._keep_size = keep_size
        self._progress = progress

    def __iter__(self) -> Iterator[LabeledImageSample]:
        return self._restorer.restore_stream(
            iter(self._source),
            batch_size=self._batch_size,
            keep_size=self._keep_size,
            progress=self._progress,
        )

    def __len__(self) -> int:
        return len(self._source)  # type: ignore[arg-type]

    def warm_cache(self) -> Any:
        return self._source.warm_cache()  # type: ignore[attr-defined]

    def __repr__(self) -> str:
        return f"RestoredStream({self._source!r} -> {self._restorer.name})"


def restore_dataset(
    source: Iterable[LabeledImageSample],
    restorer: "AutoencoderRestorerImpl | None" = None,
    *,
    batch_size: int = 16,
    keep_size: bool = True,
    progress: bool = False,
) -> RestoredStream:
    """Wrap ``source`` so its images are restored on the fly. ``restorer``
    defaults to `AutoencoderRestorer.use_default()`."""
    restorer = restorer or AutoencoderRestorerImpl.use_default()
    return RestoredStream(
        source, restorer, batch_size=batch_size, keep_size=keep_size, progress=progress
    )
