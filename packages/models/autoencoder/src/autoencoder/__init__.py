"""Autoencoder for reversing augmentation damage on images.

The training contract here is built around paired samples of the form
`ImagePairSample(input_image, target_image)`, which matches the dataset builder
used for augmentation-reversal tasks.
"""

from autoencoder.restorer import AutoencoderRestorer
from autoencoder.trainer import AutoencoderTrainer

__all__ = ["AutoencoderRestorer", "AutoencoderTrainer"]
