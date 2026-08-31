#!/usr/bin/env python
"""Project entrypoint — run the ensemble over a folder of images.

Scores every image in a directory (recursively) with the AIGC-detection
ensemble (fusion + clip_vit_b32 + dinov2 + yolo + swin) and writes
a JSON file of ``{"image_path", "pred"}`` records, where ``pred`` is
P(AI-generated) in [0, 1].

    python predict.py /path/to/images -o predictions.json

    # weighted / meta combiner, custom paths, subset of members:
    python predict.py imgs --method weighted --opensdi-repo /content/OpenSDI \
        --dino-checkpoint /content/dino.pt
    python predict.py imgs --meta-file ensemble_meta.json
    python predict.py imgs --members clip_vit_b32,dinov2,yolo,swin   # skip fusion

Prereqs: the member checkpoints in place (see each package's ``use_default`` /
``$*_CHECKPOINT``), and for the fusion member ``opensdi_detector.setup_opensdi()``
run once (or pass ``--no-fusion``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".gif"}
_ALL_MEMBERS = ("fusion", "clip_vit_b32", "dinov2", "yolo", "swin")


def _iter_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
            yield path


def _build_detector(args: argparse.Namespace):
    from ensemble import EnsembleDetector, build_default_ensemble_members

    include = [m.strip() for m in args.members.split(",") if m.strip()] if args.members else list(_ALL_MEMBERS)
    if args.no_fusion and "fusion" in include:
        include.remove("fusion")

    member_kwargs: dict[str, dict[str, Any]] = {}
    if args.dino_checkpoint:
        member_kwargs["dinov2"] = {"checkpoint": args.dino_checkpoint}
    if args.swin_checkpoint:
        member_kwargs["swin"] = {"checkpoint": args.swin_checkpoint}

    common = dict(
        device=args.device,
        include=include,
        use_autoencoder=args.use_autoencoder,
        autoencoder_checkpoint=args.autoencoder_checkpoint,
        opensdi_repo_dir=args.opensdi_repo,
        member_kwargs=member_kwargs or None,
    )

    if args.meta_file:
        from ensemble import EnsembleTrainer

        members = build_default_ensemble_members(**common)
        trainer = EnsembleTrainer.load(args.meta_file, members=members)
        det = trainer.as_detector(
            method="meta",
            decision_threshold=args.threshold,   # None -> the fitted meta threshold
        )
    else:
        det = EnsembleDetector.use_default(method=args.method, **common)
        if args.threshold is not None:
            det.decision_threshold = float(args.threshold)

    return det


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image_dir", type=Path, help="folder of images (searched recursively)")
    parser.add_argument("-o", "--output", type=Path, default=Path("predictions.json"),
                        help="output JSON file (default: predictions.json)")
    parser.add_argument("--device", default="auto", help="'auto' / 'cpu' / 'cuda' / 'cuda:0'")
    parser.add_argument("--method", default="max", choices=["max", "mean", "weighted", "meta"],
                        help="ensemble combiner (default: max). 'weighted'/'meta' use the "
                             "values baked into EnsembleDetector.use_default unless --meta-file is given.")
    parser.add_argument("--meta-file", type=Path, default=None,
                        help="a saved EnsembleTrainer bundle (json + .meta.pkl) -> method='meta'")
    parser.add_argument("--threshold", type=float, default=None,
                        help="override the decision threshold (only affects the 'verdict' field)")
    parser.add_argument("--members", default=None,
                        help="comma-separated subset of: " + ",".join(_ALL_MEMBERS))
    parser.add_argument("--no-fusion", action="store_true",
                        help="drop the fusion member (skips the OpenSDI dependency)")
    parser.add_argument("--opensdi-repo", default=None, help="path to the iamwangyabin/OpenSDI clone")
    parser.add_argument("--dino-checkpoint", default=None, help="path to dino.pt")
    parser.add_argument("--swin-checkpoint", default=None, help="path to the swin .pth")
    parser.add_argument("--use-autoencoder", action="store_true",
                        help="restore each image with the autoencoder before the fusion member")
    parser.add_argument("--autoencoder-checkpoint", default=None)
    parser.add_argument("--limit", type=int, default=None, help="only score the first N images")
    parser.add_argument("--verdict", action="store_true",
                        help="also write a 0/1 'verdict' field (pred >= threshold)")
    parser.add_argument("--indent", type=int, default=2, help="output JSON indent (0 = compact)")
    args = parser.parse_args(argv)

    if not args.image_dir.is_dir():
        parser.error(f"{args.image_dir} is not a directory")

    images = list(_iter_images(args.image_dir))
    if args.limit:
        images = images[: args.limit]
    if not images:
        parser.error(f"no images ({', '.join(sorted(_IMAGE_SUFFIXES))}) found under {args.image_dir}")
    print(f"[predict] {len(images)} images under {args.image_dir}")

    from PIL import Image

    print("[predict] building the ensemble (members load their weights on first use)...")
    det = _build_detector(args)
    print(f"[predict] detector: {det.name}  method={det.method}  threshold={det.decision_threshold}")
    print(f"[predict] members: {[getattr(m, 'name', '?') for m in det.members]}")

    try:
        from tqdm.auto import tqdm

        images = tqdm(images, unit="img", desc="scoring")
    except ImportError:
        pass

    records: list[dict[str, Any]] = []
    errors = 0
    try:
        for path in images:
            try:
                with Image.open(path) as im:
                    result = det.predict(im.convert("RGB"))
                score = float(result.ai_generated_probability)
                rec: dict[str, Any] = {"image_path": str(path), "pred": round(score, 6)}
                if args.verdict:
                    rec["verdict"] = int(score >= det.decision_threshold)
                records.append(rec)
            except Exception as error:  # noqa: BLE001 - one bad image shouldn't kill the run
                errors += 1
                print(f"\n[predict] FAILED {path}: {error}", file=sys.stderr)
                records.append({"image_path": str(path), "pred": None, "error": str(error)})
    except KeyboardInterrupt:
        print("\n[predict] interrupted — writing partial results", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(records, indent=args.indent or None), encoding="utf-8"
    )

    scored = [r["pred"] for r in records if r["pred"] is not None]
    flagged = sum(p >= det.decision_threshold for p in scored)
    print(f"\n[predict] wrote {len(records)} records to {args.output} "
          f"({errors} errors)")
    if scored:
        print(f"[predict] mean pred {sum(scored) / len(scored):.3f} | "
              f"{flagged}/{len(scored)} >= {det.decision_threshold} (AI-generated)")
    return 1 if errors and not scored else 0


if __name__ == "__main__":
    raise SystemExit(main())
