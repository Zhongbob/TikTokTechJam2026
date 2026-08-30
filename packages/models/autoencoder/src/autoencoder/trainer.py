"""Training code for an image-to-image autoencoder that learns to reverse
augmentation steps applied to photos.

The model operates on paired samples from `shared_types.ImagePairSample`:
`input_image` is the augmented/noisy/transformed image and `target_image` is
what the network should reconstruct.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from shared_types import AutoencoderTrainableModel, ImagePairSample, TrainingResult


class PairedImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Simple dataset that turns PIL image pairs into tensors."""

    def __init__(self, samples: Iterable[ImagePairSample], image_size: int = 224) -> None:
        self.samples = list(samples)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        input_tensor = self._to_tensor(sample.input_image)
        target_tensor = self._to_tensor(sample.target_image)
        return input_tensor, target_tensor

    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        rgb = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BICUBIC)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
        return torch.from_numpy(array.transpose(2, 0, 1))


class SimpleImageAutoencoder(nn.Module):
    """Small convolutional encoder-decoder for restoring damaged images."""

    def __init__(self, in_channels: int = 3, hidden_channels: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels, hidden_channels * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_channels * 2, hidden_channels * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_channels * 4, hidden_channels * 2, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_channels * 2, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class AutoencoderTrainer(AutoencoderTrainableModel):
    """Simple augmentation-reversal trainer.

    Example:
        from data.dataset_builder import load_manifest_as_samples
        trainer = AutoencoderTrainer(image_size=224)
        samples = load_manifest_as_samples("outputs/augmented")
        result = trainer.train(samples, epochs=20, batch_size=16)
        trainer.save("autoencoder.pt")
    """

    name = "autoencoder-augmentation-reversal"

    def __init__(self, image_size: int = 224, hidden_channels: int = 32, device: str = "cpu") -> None:
        self.image_size = image_size
        self.hidden_channels = hidden_channels
        self.device = torch.device(device)
        self._model: nn.Module | None = None

    def train(
        self,
        samples: Iterable[ImagePairSample],
        *,
        val_samples: Iterable[ImagePairSample] | None = None,
        val_fraction: float = 0.1,
        output_dir: str | Path = "autoencoder_runs",
        epochs: int = 20,
        batch_size: int = 16,
        learning_rate: float = 1e-3,
        device: str | None = None,
        **kwargs: Any,
    ) -> TrainingResult:
        if device is not None:
            self.device = torch.device(device)

        sample_list = list(samples)
        if not sample_list:
            raise ValueError("samples must not be empty")

        if val_samples is None:
            if len(sample_list) < 2:
                raise ValueError("Need at least 2 paired samples when val_samples is not supplied")
            split_index = max(1, int(len(sample_list) * (1 - val_fraction)))
            train_samples = sample_list[:split_index]
            val_samples = sample_list[split_index:]
        else:
            train_samples = sample_list
            val_samples = list(val_samples)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._model = SimpleImageAutoencoder(hidden_channels=self.hidden_channels).to(self.device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        train_loader = DataLoader(
            PairedImageDataset(train_samples, image_size=self.image_size),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            PairedImageDataset(val_samples, image_size=self.image_size),
            batch_size=batch_size,
            shuffle=False,
        )

        best_val_loss = float("inf")
        checkpoint_path = output_dir / "autoencoder_best.pt"
        train_loss = 0.0

        for epoch in range(1, epochs + 1):
            self._model.train()
            running_loss = 0.0
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device, dtype=torch.float32)
                targets = targets.to(self.device, dtype=torch.float32)

                optimizer.zero_grad(set_to_none=True)
                outputs = self._model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            train_loss = running_loss / max(len(train_samples), 1)
            val_loss = self._evaluate_loader(val_loader, criterion)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save(checkpoint_path)

            if epoch % max(1, epochs // 10) == 0 or epoch == epochs:
                print(f"epoch={epoch}/{epochs} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        if self._model is None:
            raise RuntimeError("Training finished without creating a model")

        last_checkpoint = output_dir / "autoencoder_last.pt"
        self.save(last_checkpoint)

        return TrainingResult(
            epochs_completed=epochs,
            final_loss=float(best_val_loss if best_val_loss != float("inf") else train_loss),
            metrics={
                "train_loss": float(train_loss),
                "val_loss": float(best_val_loss if best_val_loss != float("inf") else train_loss),
            },
            checkpoint_path=str(checkpoint_path),
            notes=f"Trained on {len(train_samples)} paired samples and validated on {len(val_samples)}.",
        )

    def _evaluate_loader(self, loader: DataLoader, criterion: nn.Module) -> float:
        if self._model is None:
            raise RuntimeError("Call train() or load() before evaluation")

        self._model.eval()
        total_loss = 0.0
        total_count = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device, dtype=torch.float32)
                targets = targets.to(self.device, dtype=torch.float32)
                outputs = self._model(inputs)
                total_loss += criterion(outputs, targets).item() * inputs.size(0)
                total_count += inputs.size(0)
        return total_loss / max(total_count, 1)

    def evaluate(self, samples: Iterable[ImagePairSample], **kwargs: Any) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("Call train() or load() before evaluate()")

        batch_size = int(kwargs.pop("batch_size", 16))
        loader = DataLoader(PairedImageDataset(samples, image_size=self.image_size), batch_size=batch_size, shuffle=False)
        criterion = nn.MSELoss()
        mse = self._evaluate_loader(loader, criterion)
        return {"mse": float(mse), "rmse": float(np.sqrt(mse))}

    def save(self, path: str | Path) -> None:
        if self._model is None:
            raise RuntimeError("Nothing trained yet — call train() first")

        serialized = {
            "model_state": self._model.state_dict(),
            "image_size": self.image_size,
            "hidden_channels": self.hidden_channels,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(serialized, path)

    @classmethod
    def load(cls, path: str | Path) -> "AutoencoderTrainer":
        checkpoint = torch.load(path, map_location="cpu")
        trainer = cls(image_size=int(checkpoint.get("image_size", 224)), hidden_channels=int(checkpoint.get("hidden_channels", 32)))
        trainer._model = SimpleImageAutoencoder(hidden_channels=trainer.hidden_channels)
        trainer._model.load_state_dict(checkpoint["model_state"])
        return trainer


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a simple autoencoder to reverse image augmentations.")
    parser.add_argument("--data-dir", type=str, default="outputs/augmented", help="Directory containing paired inputs/targets from AutoencoderDatasetBuilder")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str, default="autoencoder_runs")
    parser.add_argument("--device", type=str, default="cpu")
    return parser


def _run_cli(argv: list[str] | None = None) -> None:
    args = _build_cli_parser().parse_args(argv)

    from data.dataset_builder.autoencoder import load_manifest_as_samples

    samples = load_manifest_as_samples(args.data_dir)
    trainer = AutoencoderTrainer(image_size=args.image_size, device=args.device)
    result = trainer.train(
        samples,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    print("=== Training result ===")
    print(f"epochs_completed : {result.epochs_completed}")
    print(f"final_loss       : {result.final_loss}")
    print(f"checkpoint_path  : {result.checkpoint_path}")
    print(f"notes            : {result.notes}")


if __name__ == "__main__":
    _run_cli()
