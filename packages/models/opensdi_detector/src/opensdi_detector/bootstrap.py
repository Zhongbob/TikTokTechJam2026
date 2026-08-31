"""One-shot environment setup for `OpenSDIDetector`.

`OpenSDIDetector` can't run from a bare `pip install` — MaskCLIP's model code
is not vendored. This module does the rest:

  1. clone ``iamwangyabin/OpenSDI``            -> ``<repo_root>/OpenSDI``
  2. pip-install its deps (``IMDLBenCo``, ``timm``, OpenAI ``clip``, ...)
  3. download the MaskCLIP checkpoint from HF  -> ``<weights_dir>/``
  4. download the MAE ViT-B pretrain weights   -> ``<weights_dir>/`` (and into
     ``./weights/`` — OpenSDI's ``clip_utils.py`` hard-codes that relative path)
  5. pre-download CLIP ViT-L/14

Colab, one line::

    from opensdi_detector import setup_opensdi
    info = setup_opensdi()                      # repo -> /content/OpenSDI, weights -> package
    det  = OpenSDIDetector.use_default(repo_dir=info["repo_dir"])

or from a shell::

    python -m opensdi_detector.bootstrap --repo-root /content

Defaults: repo into ``/content/OpenSDI``, weights into this package's
``src/weights/`` folder.

⚠️  ``IMDLBenCo`` pins ``numpy<2`` and ``albumentations==1.3.0`` — installing it
may downgrade NumPy in the runtime. Restart the runtime afterwards if other
libraries complain.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from opensdi_detector.detector import DEFAULT_WEIGHTS_DIR

OPENSDI_REPO_URL = "https://github.com/iamwangyabin/OpenSDI.git"
DEFAULT_REPO_ROOT = "/content"

#: Newest MaskCLIP checkpoint in the HF weights repo (SD1.5 training split — the
#: only one OpenSDI released training data for).
DEFAULT_CHECKPOINT_FILE = "MaskCLIP_sd15_20241109_08_53_19.pth"
#: Name `OpenSDIDetector.use_default()` looks for — bootstrap also links to this.
USE_DEFAULT_ALIAS = "maskclip_opensdi.pth"
HF_WEIGHTS_REPO = "nebula/MaskCLIP-weights"

MAE_URL = "https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth"
MAE_FILE = "mae_pretrain_vit_base.pth"

#: Enough to import ``model.MaskCLIP`` and run inference (not the full training
#: stack in requirements.txt).
_MINIMAL_DEPS = ["IMDLBenCo", "timm", "ftfy", "regex", "omegaconf"]
_CLIP_SPEC = "git+https://github.com/openai/CLIP.git"


def _run(cmd: list[str]) -> None:
    print("  + " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def _pip_install(*args: str) -> None:
    _run([sys.executable, "-m", "pip", "install", "-q", *args])


def _clone_repo(repo_dir: Path, force: bool) -> Path:
    if repo_dir.exists() and not force:
        if (repo_dir / "model" / "MaskCLIP.py").is_file():
            print(f"[opensdi-setup] repo already present at {repo_dir}")
            return repo_dir
        raise FileNotFoundError(
            f"{repo_dir} exists but doesn't look like the OpenSDI repo — "
            "remove it or pass force=True."
        )
    if force and repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "clone", "--depth", "1", OPENSDI_REPO_URL, str(repo_dir)])
    return repo_dir


def _install_deps(repo_dir: Path, full_requirements: bool, force: bool) -> None:
    if not force:
        try:
            import clip  # noqa: F401
            import IMDLBenCo  # noqa: F401

            print("[opensdi-setup] deps already importable — skipping (force=True to redo)")
            return
        except ImportError:
            pass
    print("[opensdi-setup] installing dependencies "
          "(IMDLBenCo pins numpy<2 — a NumPy downgrade here is expected)")
    if full_requirements:
        _pip_install("-r", str(repo_dir / "requirements.txt"))
    else:
        _pip_install(*_MINIMAL_DEPS)
    _pip_install(_CLIP_SPEC)


def _download_url(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"[opensdi-setup] already have {dest.name} "
              f"({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    import torch

    print(f"[opensdi-setup] downloading {url}\n              -> {dest}")
    torch.hub.download_url_to_file(url, str(dest))
    return dest


def _download_hf(repo_id: str, filename: str, dest_dir: Path) -> Path:
    dest = dest_dir / filename
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"[opensdi-setup] already have {filename} "
              f"({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        got = hf_hub_download(repo_id=repo_id, filename=filename,
                              local_dir=str(dest_dir))
        return Path(got)
    except Exception as error:  # noqa: BLE001 - fall back to a plain URL
        print(f"[opensdi-setup] huggingface_hub unavailable ({error}); using direct URL")
        return _download_url(
            f"https://huggingface.co/{repo_id}/resolve/main/{filename}", dest
        )


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)  # hardlink — free, no extra disk
    except OSError:
        shutil.copy(src, dst)


def setup_opensdi(
    repo_root: str | os.PathLike[str] = DEFAULT_REPO_ROOT,
    weights_dir: str | os.PathLike[str] | None = None,
    *,
    checkpoint_file: str = DEFAULT_CHECKPOINT_FILE,
    hf_weights_repo: str = HF_WEIGHTS_REPO,
    install_deps: bool = True,
    full_requirements: bool = False,
    place_mae_in_cwd: bool = True,
    prewarm_clip: bool = True,
    force: bool = False,
) -> dict[str, str]:
    """Provision everything `OpenSDIDetector` needs.

    Args:
        repo_root: OpenSDI is cloned to ``<repo_root>/OpenSDI`` (default
            ``/content`` — the Colab working dir).
        weights_dir: where checkpoints are downloaded (default: this package's
            ``src/weights/`` folder, which is also where ``use_default()`` looks).
        checkpoint_file: which file to pull from the HF weights repo.
        hf_weights_repo: the HF repo id holding the MaskCLIP ``.pth`` files.
        install_deps: run the pip installs (set False if you manage deps yourself).
        full_requirements: ``pip install -r OpenSDI/requirements.txt`` (pinned,
            heavier) instead of the minimal inference set.
        place_mae_in_cwd: also drop the MAE weights at ``./weights/`` — OpenSDI's
            ``clip_utils.py`` ``torch.load``s that exact relative path.
        prewarm_clip: pre-download CLIP ViT-L/14 (~1 GB) now rather than on first
            ``predict()``.
        force: re-clone and re-install even if things look present.

    Returns:
        ``{"repo_dir", "weights_dir", "checkpoint_path", "checkpoint_alias",
        "mae_weights"}`` — all absolute path strings. Also sets
        ``os.environ["OPENSDI_REPO"]``.
    """
    repo_dir = (Path(repo_root).expanduser() / "OpenSDI").resolve()
    wdir = (
        Path(weights_dir).expanduser().resolve()
        if weights_dir is not None
        else DEFAULT_WEIGHTS_DIR
    )
    wdir.mkdir(parents=True, exist_ok=True)

    print(f"[opensdi-setup] repo    -> {repo_dir}")
    print(f"[opensdi-setup] weights -> {wdir}\n")

    _clone_repo(repo_dir, force)
    if install_deps:
        _install_deps(repo_dir, full_requirements, force)

    checkpoint_path = _download_hf(hf_weights_repo, checkpoint_file, wdir)
    alias_path = wdir / USE_DEFAULT_ALIAS
    _link_or_copy(checkpoint_path, alias_path)

    mae_path = _download_url(MAE_URL, wdir / MAE_FILE)
    if place_mae_in_cwd:
        cwd_mae = Path.cwd() / "weights" / MAE_FILE
        _link_or_copy(mae_path, cwd_mae)
        print(f"[opensdi-setup] MAE weights also at {cwd_mae} "
              "(OpenSDI hard-codes weights/mae_pretrain_vit_base.pth)")

    if prewarm_clip:
        try:
            import clip

            print("[opensdi-setup] pre-downloading CLIP ViT-L/14 ...")
            clip.load("ViT-L/14", device="cpu")
        except Exception as error:  # noqa: BLE001
            print(f"[opensdi-setup] CLIP prewarm skipped ({error})")

    os.environ["OPENSDI_REPO"] = str(repo_dir)

    result = {
        "repo_dir": str(repo_dir),
        "weights_dir": str(wdir),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_alias": str(alias_path),
        "mae_weights": str(mae_path),
    }

    print(
        f"""
[opensdi-setup] done.

    from opensdi_detector import OpenSDIDetector
    det = OpenSDIDetector.use_default(
        repo_dir=r"{repo_dir}",
        score_mode="mask", mask_reduce="max",
    )

  (OPENSDI_REPO is set for this process, so repo_dir= can be omitted above.)
"""
    )
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Clone the OpenSDI repo + fetch weights so OpenSDIDetector can run."
    )
    parser.add_argument("--repo-root", default=DEFAULT_REPO_ROOT,
                        help="clone into <repo-root>/OpenSDI (default: %(default)s)")
    parser.add_argument("--weights-dir", default=None,
                        help="download weights here (default: the opensdi_detector "
                             "package's src/weights/ folder)")
    parser.add_argument("--checkpoint-file", default=DEFAULT_CHECKPOINT_FILE,
                        help="file to pull from the HF weights repo (default: %(default)s)")
    parser.add_argument("--hf-weights-repo", default=HF_WEIGHTS_REPO)
    parser.add_argument("--no-deps", action="store_true", help="skip all pip installs")
    parser.add_argument("--full-requirements", action="store_true",
                        help="pip install -r OpenSDI/requirements.txt (pinned, heavier)")
    parser.add_argument("--no-mae-in-cwd", action="store_true",
                        help="don't copy MAE weights into ./weights/")
    parser.add_argument("--no-prewarm-clip", action="store_true",
                        help="don't pre-download CLIP ViT-L/14")
    parser.add_argument("--force", action="store_true",
                        help="re-clone and re-install even if present")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    setup_opensdi(
        repo_root=args.repo_root,
        weights_dir=args.weights_dir,
        checkpoint_file=args.checkpoint_file,
        hf_weights_repo=args.hf_weights_repo,
        install_deps=not args.no_deps,
        full_requirements=args.full_requirements,
        place_mae_in_cwd=not args.no_mae_in_cwd,
        prewarm_clip=not args.no_prewarm_clip,
        force=args.force,
    )


if __name__ == "__main__":
    main()
