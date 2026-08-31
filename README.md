# TikTokTechJam 2026 — AI-Generated Image Detection

Detects AI-generated images, robust to common real-world transformations
(JPEG compression, blur, resize, noise, colour jitter, cropping).

The model is an **ensemble** that combines five detectors:

| member | what it is |
|---|---|
| `fusion` | Community-Forensics (whole-image synthetic) **+** OpenSDI/MaskCLIP (diffusion-inpaint localizer), max-combined |
| `clip_vit_b32` | fine-tuned OpenAI CLIP ViT-B/32, prompt-similarity head |
| `dinov2` | fine-tuned DINOv2 backbone + linear head |
| `yolo` | Ultralytics YOLO classification fine-tuned on SID-Set |
| `swin` | fine-tuned Swin-Tiny |

Their per-image `p(AI)` scores are combined with `max`, a fitted linear
`weighted` rule, or a tree-based `meta` classifier. Optionally an
**autoencoder** restores each image before the `fusion` member sees it (the
other members always get the original).

The repo is one [uv](https://docs.astral.sh/uv/) workspace: many small packages
that share a lockfile/environment.

---

## Quick start — score a folder of images

```powershell
# 1. install uv (once)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. install every workspace package into one environment
uv sync --all-packages

# 3. (for the fusion member) clone OpenSDI + fetch its weights, once
uv run --package opensdi_detector python -m opensdi_detector.bootstrap --repo-root .

# 4. run the ensemble over an image directory
uv run --package ensemble python predict.py path/to/images -o predictions.json
```

`predict.py` writes a JSON list of `{"image_path", "pred"}`, where `pred` is
`P(AI-generated)` in `[0, 1]`:

```json
[
  {"image_path": "path/to/images/a.png", "pred": 0.9124},
  {"image_path": "path/to/images/b.jpg", "pred": 0.0481}
]
```

### `predict.py` options

| flag | |
|---|---|
| `--method {max,mean,weighted,meta}` | combiner (default `max`) |
| `--meta-file ensemble_meta.json` | load a fitted `EnsembleTrainer` bundle → `method="meta"` |
| `--members clip_vit_b32,dinov2,yolo,swin` / `--no-fusion` | run a subset |
| `--opensdi-repo`, `--dino-checkpoint`, `--swin-checkpoint` | checkpoint paths |
| `--device`, `--threshold`, `--use-autoencoder`, `--limit`, `--verdict` | |

`python predict.py --help` for everything.

### Checkpoints

Each detector loads its own weights on first use. In priority order:
`checkpoint=` / `--*-checkpoint`, then `$<PKG>_CHECKPOINT`, then the package's
`src/weights/` folder, the repo root, the cwd, `/content`. Files over ~100 MB
are **not** committed. `dino.pt` (~253 MB) is fetched automatically from the
project's Hugging Face bucket (`Zhongbob2/TikTokTechJam`) into
`packages/models/dinov2/src/weights/` when it isn't found locally; override with
a path or `$DINOV2_CHECKPOINT`.

---

## Repository layout

```text
predict.py                Main entrypoint: score an image folder -> predictions.json
apps/
  web/                    Streamlit demo UI (upload -> transform -> restore -> detect)
packages/
  data/                   SID-Set streaming + augmentation + WildFake eval dataset
  models/
    ensemble/             THE MODEL: fusion + clip + dinov2 + yolo + swin, combined
    fusion/               Community-Forensics + OpenSDI sub-combiner
    community_forensics/  fusion member (HF ViT)
    opensdi_detector/     fusion member (MaskCLIP) + bootstrap script
    clip_vit_b32/ dinov2/ yolo/ swin/   trained single-model detectors (ensemble members)
    autoencoder/          augmentation-reversal restoration model + trainer
libs/
  detector_common/        ImageDetector base, CombinerDetector/CombinerTrainer, MetaClassifier,
                          checkpoint resolution
  shared_types/           shared dataclasses / Protocols
  image_io/               image loading helpers
```

Every package has its own `pyproject.toml` and is a `[tool.uv.workspace]`
member. `uv sync --all-packages` installs them all; or `cd <package>` +
`uv sync` for just one.

---

## Training the combiner

The `max` combiner needs no training. `weighted` and `meta` are fitted with
`ensemble.EnsembleTrainer` on labelled data (augmented SID-Set):

```python
from ensemble import EnsembleTrainer

tr = EnsembleTrainer.use_default(opensdi_repo_dir="OpenSDI",
                                 member_kwargs={"dinov2": {"checkpoint": "dino.pt"}})

Xtr, ytr = tr.member_score_matrix(train_samples)   # runs all 5 members once — cache this
Xva, yva = tr.member_score_matrix(val_samples)

tr.compare_methods(X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva,
                   meta_kinds=("tree", "gboost"))   # max vs weighted vs meta

tr.fit_meta_classifier(X=Xtr, y=ytr, kind="gboost")  # or tr.optimal_weights(X=Xtr, y=ytr)
tr.save("ensemble_meta.json")                        # -> reuse with predict.py --meta-file
```

Training data lives in `packages/data` — `augmented_sid_dataset(...)` (SID-Set,
streamed + augmented) and `eval_dataset(...)` (the WildFake benchmark).
**Evaluation is under augmentation** — that's the scored metric.

---

## Web demo

```powershell
cd apps/web
uv sync
uv run streamlit run src/web/main.py     # http://localhost:8501
```

Pick **YOLO Classifier** (direct) or **Ensemble (Transform Reversal)** (the full
ensemble; shows the autoencoder-restored image the fusion sub-model scores),
choose an image, apply real-world transforms, see the verdict.

---

## Common uv commands

| Command | What it does |
|---|---|
| `uv sync --all-packages` | Install every workspace package into the shared environment |
| `uv run --package <name> <cmd>` | Run `<cmd>` with `<name>`'s dependencies available |
| `uv run --package <name> pytest` | Run a package's tests |
| `uv add <dep>` | Add a dependency to the current package (run from its folder) |
