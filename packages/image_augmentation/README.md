# Image Augmentation Package

This folder is designed to merge into the existing `TikTokTechJam2026`
monorepo under `packages/image_augmentation/`.

It provides:

- all-five randomized augmentation permutations;
- local PNG/JPEG folder augmentation;
- paired clean/augmented autoencoder dataset generation;
- balanced streaming subsets from `saberzl/SID_Set`;
- offline tests and example scripts.

Shared Python types are intentionally located at:

```text
libs/shared_types/augmentation.py
```

## Dependencies

Add the following dependencies to the repository's existing root
`pyproject.toml`. Do not replace that file.

```toml
"datasets>=3.0,<5",
"numpy>=1.26,<3",
"Pillow>=10,<13",
```

Add `pytest>=8,<10` only if the repository does not already have a test
dependency.

## Run a small SID-Set test

From the repository root:

```bash
python -m packages.image_augmentation.cli \
  --images-per-label 2 \
  --output-dir outputs/sid_test \
  --size 256 \
  --seed 4
```

This holds retrieved source images in memory and saves only the required
autoencoder inputs, targets, and manifest.

## Run tests

From the repository root:

```bash
pytest packages/image_augmentation/tests
```

## Generated data

Ensure the existing root `.gitignore` contains:

```gitignore
outputs/
content/
data/
```

Do not commit generated image datasets.

