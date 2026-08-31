"""Stream the WildFake eval subset (``techjam-aigc/wildfake-eval-subset``).

Reference benchmark for the AIGC-detection track: real (COCO / LAION) vs
diffusion-generated images, repackaged as parquet. ``eval_dataset()`` returns
the same re-iterable, memory-bounded, optionally-augmented stream as
`augmented_sid_dataset()`.

Configs (all ``split="validation"``):

* ``default``          -- 4,998 ``coco_val2017`` (real) + 8,843 ``dalle3_advanced``
                          (fake). **Original bytes, untouched.** Real images are
                          all 200x200; fakes vary -> pass ``output_size=`` for a
                          fair eval, or use ``normalized``.
* ``normalized``       -- same images, centre-cropped + resized to 200 + JPEG q92.
* ``laion_matched``    -- 3,826 LAION + 3,826 DALL-E 3, both downscaled to 512.
* ``cross_generator``  -- 1,500 LAION vs DALL-E 3 / Midjourney v5 / SDXL / GigaGAN,
                          all downscaled to 256.

None of the configs apply the six problem-statement corruptions, so this is a
*clean* benchmark by default. Pass ``augment=True`` to run the same
`ImageAugmenter` pipeline `augmented_sid_dataset` uses.

⚠️  The repo is **private to the ``techjam-aigc`` org** -- ``hf auth login`` (or
pass ``hf_token=``) first, or ``load_dataset`` 401s.

⚠️  Per the dataset card: **evaluation only, do not train on it** -- the final
test set is drawn from the same corpus, so fitting anything (a classifier, a
meta-classifier, even picking a threshold to keep) on these images leaks.
"""

from __future__ import annotations

import math
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Iterator, Sequence

from shared_types import SourceMetadata

from .streaming import StreamingAugmentedDataset

#: cap spec passed around: a single int, a per-source ``{source: int}`` map, or None.
CapSpec = "int | Mapping[str, int] | None"


def _cap_for(source: str, spec: "int | Mapping[str, int] | None") -> int | None:
    if spec is None:
        return None
    if isinstance(spec, Mapping):
        return spec.get(source)
    return int(spec)

WILDFAKE_REPO = "techjam-aigc/wildfake-eval-subset"

#: label id -> name. WildFake fakes are fully generated (not locally tampered),
#: so "synthetic" matches SID's `label_name` vocabulary.
WILDFAKE_LABEL_NAMES = {0: "real", 1: "synthetic"}

#: Per-(config, source) example counts, straight from the dataset card -- used to
#: size the progress bar / ``__len__`` without a counting pass.
WILDFAKE_SOURCE_COUNTS: dict[str, dict[str, int]] = {
    "default": {"coco_val2017": 4998, "dalle3_advanced": 8843},
    "normalized": {"coco_val2017": 4998, "dalle3_advanced": 8843},
    "laion_matched": {"laion5b": 3826, "dalle3": 3826},
    "cross_generator": {
        "laion5b": 1500, "dalle3": 1000, "midjourney_v5": 999,
        "sdxl": 1000, "gigagan": 995,
    },
}

#: None of the published configs apply the problem-statement corruptions.
WILDFAKE_IS_AUGMENTED = False


def iter_wildfake_encoded(
    config: str = "default",
    split: str = "validation",
    *,
    sources: Sequence[str] | None = None,
    max_per_source: "int | Mapping[str, int] | None" = None,
    shuffle: bool = False,
    seed: int = 4,
    buffer_size: int = 100,
    hf_token: str | None = None,
) -> Iterator[tuple[bytes, SourceMetadata]]:
    """Yield ``(raw encoded image bytes, metadata)`` from the WildFake eval subset.

    metadata keys: ``img_id``, ``binary_aigc_label`` (0 real / 1 fake),
    ``label_name`` ("real" / "synthetic"), ``source`` (e.g. ``"coco_val2017"``,
    ``"dalle3_advanced"``).

    Args:
        config: one of ``WILDFAKE_SOURCE_COUNTS`` (``"default"``, ``"normalized"``,
            ``"laion_matched"``, ``"cross_generator"``).
        sources: keep only these ``source`` values; ``None`` keeps all.
        max_per_source: cap emitted images per ``source`` -- a single int (same
            cap for every source), a per-source ``{source: int}`` map, or ``None``
            for no cap.
        shuffle: stream through a shuffle buffer (default False -- benchmark
            order is deterministic anyway).
    """
    if hf_token is not None:
        os.environ["HF_TOKEN"] = hf_token

    from datasets import Image as _HFImage
    from datasets import load_dataset

    dataset = load_dataset(WILDFAKE_REPO, name=config, split=split, streaming=True)
    dataset = dataset.cast_column("image", _HFImage(decode=False))
    if shuffle:
        dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)

    wanted = {str(s) for s in sources} if sources is not None else None

    def _all_wanted_capped(counts: Counter[str]) -> bool:
        if wanted is None:
            return False  # other sources may still appear
        for s in wanted:
            cap = _cap_for(s, max_per_source)
            if cap is None or counts[s] < cap:
                return False
        return True

    counts: Counter[str] = Counter()
    emitted = 0
    for example in dataset:
        source = str(example.get("source", "unknown"))
        if wanted is not None and source not in wanted:
            continue
        cap = _cap_for(source, max_per_source)
        if cap is not None and counts[source] >= cap:
            if _all_wanted_capped(counts):
                break
            continue

        entry = example["image"]
        data = entry["bytes"]
        if data is None:  # feature backed by a local file rather than inline bytes
            data = Path(entry["path"]).read_bytes()

        label = int(example["label"])
        metadata: SourceMetadata = {
            "img_id": str(example.get("id") or f"{config}-{emitted}"),
            "binary_aigc_label": label,
            "label_name": WILDFAKE_LABEL_NAMES.get(label, str(label)),
            "source": source,
        }
        counts[source] += 1
        emitted += 1
        yield data, metadata


def _resolve_caps(
    config: str,
    chosen: Sequence[str],
    max_per_source: "int | Mapping[str, int] | None",
    fraction: float | None,
) -> "int | Mapping[str, int] | None":
    """Turn ``fraction`` into a per-source cap map (from the known counts);
    otherwise pass ``max_per_source`` through. Rejects giving both."""
    if fraction is None:
        return max_per_source
    if max_per_source is not None:
        raise ValueError("pass either fraction= or max_per_source=, not both")
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1] (0.1 = 10% of each source)")
    known = WILDFAKE_SOURCE_COUNTS.get(config, {})
    caps: dict[str, int] = {}
    for source in chosen:
        count = known.get(source)
        if count is None:
            raise ValueError(
                f"fraction= needs known per-source counts; source {source!r} in "
                f"config {config!r} isn't in WILDFAKE_SOURCE_COUNTS -- use max_per_source="
            )
        caps[source] = max(1, math.ceil(count * fraction))
    return caps


def _resolve_selection(
    config: str,
    sources: Sequence[str] | None,
    cap_spec: "int | Mapping[str, int] | None",
) -> tuple[list[str], int]:
    """(chosen source list, expected total) for progress/`__len__`."""
    known = WILDFAKE_SOURCE_COUNTS.get(config, {})
    chosen = [str(s) for s in sources] if sources is not None else list(known)
    total = 0
    for source in chosen:
        count = known.get(source)
        cap = _cap_for(source, cap_spec)
        if count is None:
            total += cap or 0  # unknown source -> can't size precisely
        elif cap is not None:
            total += min(count, cap)
        else:
            total += count
    return chosen, total


def eval_dataset(
    config: str = "default",
    split: str = "validation",
    *,
    sources: Sequence[str] | None = None,
    fraction: float | None = None,
    max_per_source: "int | Mapping[str, int] | None" = None,
    augment: bool = False,
    num_augmentations: int | tuple[int, int] = 6,
    variants_per_image: int = 1,
    output_size: tuple[int, int] | None = None,
    backend: str = "thread",
    num_workers: int | None = None,
    prefetch: int | None = None,
    cache_dir: str | os.PathLike[str] | None = None,
    progress: bool = True,
    shuffle: bool = False,
    seed: int = 4,
    buffer_size: int = 100,
    hf_token: str | None = None,
) -> StreamingAugmentedDataset:
    """Re-iterable WildFake eval stream, same shape as `augmented_sid_dataset()`.

    Defaults to ``config="default"`` = 4,998 COCO val2017 (real) + 8,843 DALL-E 3
    (fake). It is a *benchmark* set, so ``augment=False`` by default -- pass
    ``augment=True`` to run the same six-corruption `ImageAugmenter` pipeline
    (`num_augmentations` / `variants_per_image` then behave as in
    `augmented_sid_dataset`).

    Each yielded `LabeledImageSample` has ``metadata`` with ``binary_aigc_label``
    (0 real / 1 fake), ``label_name`` ("real"/"synthetic"), ``source`` and
    ``img_id`` -- so a per-``source`` breakdown works the same way the SID
    per-``label_name`` one does.

    Subsetting (for a faster sanity-check pass):

    * ``fraction`` -- take this share of *every* source, so the class balance is
      kept (``fraction=0.1`` -> 10% of COCO + 10% of DALL-E 3). In ``(0, 1]``.
    * ``max_per_source`` -- an absolute cap: one int for all sources, or a
      ``{source: int}`` map. Mutually exclusive with ``fraction``.

    Both take the *first* N per source (deterministic). With a subset, prefer
    ``shuffle=True`` -- it makes the sample representative *and* lets the stream
    stop early instead of scanning past the un-sampled rows.

    NOTE: in ``config="default"`` the real (COCO) images are all 200x200 while the
    fakes vary -- a size shortcut. Pass ``output_size=(224, 224)`` (or use
    ``config="normalized"``) for a fair evaluation.

    ::

        val = eval_dataset(output_size=(224, 224), cache_dir="wildfake_cache")
        val.warm_cache()
        detector.evaluate(val, generate_confusion_matrix=True)

        quick = eval_dataset(fraction=0.1, output_size=(224, 224))   # ~1,384 imgs
    """
    provisional, _ = _resolve_selection(config, sources, None)
    cap_spec = _resolve_caps(config, provisional, max_per_source, fraction)
    chosen, total = _resolve_selection(config, sources, cap_spec)
    if total < 1:
        raise ValueError(
            f"can't size config={config!r} sources={sources!r} -- pass an explicit "
            "max_per_source= (and a known config)."
        )

    source_tag = "all" if sources is None else "+".join(sorted(chosen))
    if fraction is not None:
        sel_tag = f"-frac{fraction:g}"
    elif isinstance(max_per_source, Mapping):
        sel_tag = "-cap" + "_".join(f"{k}{v}" for k, v in sorted(max_per_source.items()))
    elif max_per_source is not None:
        sel_tag = f"-cap{max_per_source}"
    else:
        sel_tag = ""
    cache_key = f"wildfake-{config}-{split}-{source_tag}{sel_tag}-raw"

    # Pass the resolved source list (not the raw `sources`) so the iterator can
    # stop once every selected source has hit its cap instead of draining the
    # whole parquet -- matters when `fraction`/`max_per_source` is set.
    return StreamingAugmentedDataset(
        encoded_source=lambda: iter_wildfake_encoded(
            config, split,
            sources=chosen, max_per_source=cap_spec,
            shuffle=shuffle, seed=seed, buffer_size=buffer_size, hf_token=hf_token,
        ),
        total=total,
        cache_key=cache_key,
        desc=f"WildFake {config}",
        seed=seed,
        buffer_size=buffer_size,
        augment=augment,
        num_augmentations=num_augmentations,
        variants_per_image=variants_per_image,
        output_size=output_size,
        backend=backend,
        num_workers=num_workers,
        prefetch=prefetch,
        cache_dir=cache_dir,
        progress=progress,
    )
