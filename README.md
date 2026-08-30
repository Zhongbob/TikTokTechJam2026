# TikTokTechJam2026

AI-generated-image detection, robust to common real-world transformations
(compression, blur, resize, noise, color jitter, cropping). This repo is a
single [uv](https://docs.astral.sh/uv/) workspace containing several
projects that share a lockfile/environment but are run independently.

## Repository layout

```text
apps/
  web/                    Streamlit demo UI (upload/transform/detect pipeline)
packages/
  data/                   CLI: builds autoencoder training pairs (clean/augmented -> clean)
  models/
    autoencoder/          Restoration model (not yet implemented)
    ensemble/              Ensemble AI-detector model (not yet implemented)
    our_classifier/        Custom classifier model (not yet implemented)
libs/
  shared_types/           Shared dataclasses/enums/interfaces (not run directly)
  image_io/                Shared image loading/discovery helpers (not run directly)
```

Every project has its own `pyproject.toml` and is a member of the root
`[tool.uv.workspace]`, so `uv` resolves them all together into one shared
environment, but you `run`/`sync` **from inside the specific project's
folder** so uv knows which one is "active."

## 1. One-time setup

Install `uv` (skip if you already have it — check with `uv --version`):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Restart your terminal afterward so `uv` is on `PATH`.

## 2. Running a project

The general pattern is:

```powershell
cd <project-folder>     # the folder containing that project's pyproject.toml
uv sync                 # installs/updates that project's dependencies
uv run <command>         # runs it inside the synced environment
```

`uv sync`/`uv run` from inside a project folder only need to be run again
after you pull changes that touch dependencies — otherwise `uv run` alone is
enough, since it re-syncs automatically if anything is stale.

> Running `uv sync`/`uv run` from the **repo root** instead only installs the
> root workspace project's own dependencies (it has none), not any member's —
> that's why step 2 says `cd` into the specific project first. To stay at the
> root instead, add `--package <name>` to any `uv` command, e.g.
> `uv run --package web streamlit run apps/web/src/web/main.py`.

### apps/web — Streamlit demo UI

Not a plain script — it must be launched via `streamlit run`, not
`uv run python main.py`:

```powershell
cd apps/web
uv run streamlit run src/web/main.py
```

Opens at `http://localhost:8501`. Stop with `Ctrl+C`.

### packages/data — dataset generation CLI

An `argparse` CLI with two subcommands (`local` folder or streamed `sid`
subset). Run as a module so its internal `data.*` imports resolve:

```powershell
cd packages/data
uv sync
uv run src/data/main.py --output ../../images/output local --input ../../images/input
```

or, streaming a balanced subset from the SID-Set dataset:

```powershell
uv run python -m data.main --output outputs/sid --num-augmentations 6 --backend process --workers 4 sid --images-per-label 100 --hf-token <YOUR_HF_TOKEN>
```

See [packages/data/README.md](packages/data/README.md) for the full CLI
reference and output format.

### packages/models/{autoencoder,ensemble,our_classifier}

Scaffolding only — their `main.py` files are currently empty placeholders,
nothing to run yet.

### libs/shared_types, libs/image_io

Shared libraries consumed by the other projects (via
`[tool.uv.sources] ... = { workspace = true }`), not standalone
programs — there's nothing to `uv run` here directly.

## Common uv commands

| Command | What it does |
|---|---|
| `uv sync` | Install/update the current project's dependencies (run from its folder) |
| `uv sync --all-packages` | Install every workspace member's dependencies into the shared environment |
| `uv run <cmd>` | Run `<cmd>` inside the synced environment, auto-syncing first if needed |
| `uv run --package <name> <cmd>` | Run `<cmd>` for a specific workspace member without `cd`-ing into it |
| `uv add <dep>` | Add a dependency to the current project's `pyproject.toml` and sync it |
| `uv run pytest` | Run a project's test suite (e.g. from `apps/web`) |
