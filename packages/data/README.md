# AIGC Autoencoder Dataset Generator

Creates a 2× autoencoder dataset from PNG/JPEG images:

- `clean → clean`
- `augmented → clean`

Each augmented input receives a random ordered subset of 1–6 transformations: JPEG compression, Gaussian blur, resize down/up, Gaussian noise, colour jitter, and 80% centre crop.

## Project layout

This package lives inside the `TikTokTechJam2026` uv workspace. Types and
image-loading helpers that used to live in a local `libs/` folder here now
live in their own shared workspace libraries (`libs/shared_types` and
`libs/image_io` at the repo root), so multiple packages can depend on them:

```text
packages/data/
├── pyproject.toml
└── src/data/
    ├── main.py
    ├── augmentation/image_augmenter.py    # imports shared_types, image_io
    └── datasets/
        ├── autoencoder.py                 # imports shared_types, data.augmentation
        └── sid.py
```

## Setup

From the repo root (this package is a member of the root uv workspace, not
a standalone project):

```bash
uv sync
```

## Generate from a local folder

```bash
uv run python -m data.main --output outputs/local --num-augmentations 6 --backend process --workers 4 local --input path/to/images
```

The folder is searched recursively for `.png`, `.jpg`, and `.jpeg` files.

## Generate from SID-Set

Change `--images-per-label` to control how many images are selected from each SID label:

```bash
uv run src/data/main.py --output ../../images/output local --input ../../images/input
```

This streams the source images into memory and does not save separate copies of them. With three SID labels and `100` images per label, the result contains `300` source images and `600` autoencoder pairs.

## Parallel-processing options

- `--backend process`: recommended for normal `.py` execution and CPU-heavy augmentation.
- `--backend thread`: useful when process startup or image serialization is expensive.
- `--backend sequential`: easiest mode for debugging.
- `--workers 4`: number of concurrent workers.
- `--batch-size 32`: limits transformed images held in memory at once.

The process worker is defined at module level, and `main.py` uses an `if __name__ == "__main__"` guard so multiprocessing works safely.

## Output

```text
chosen_output_directory/
├── inputs/
│   ├── 000000_clean.png
│   ├── 000000_augmented.png
│   └── ...
├── targets/
│   ├── 000000_clean.png
│   ├── 000000_augmented.png
│   └── ...
└── manifest.json
```

The clean target is deliberately saved twice because each input requires its corresponding training target.
