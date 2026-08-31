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

import os
from collections import Counter
from pathlib import Path
from typing import Iterator, Sequence

from shared_types import SourceMetadata

from .streaming import StreamingAugmentedDataset

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
    max_per_source: int | None = None,
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
        max_per_source: cap emitted images per ``source`` (``None`` = no cap).
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
    counts: Counter[str] = Counter()
    emitted = 0
    for example in dataset:
        source = str(example.get("source", "unknown"))
        if wanted is not None and source not in wanted:
            continue
        if max_per_source is not None and counts[source] >= max_per_source:
            if wanted is not None and all(counts[s] >= max_per_source for s in wanted):
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


def _resolve_selection(
    config: str,
    sources: Sequence[str] | None,
    max_per_source: int | None,
) -> tuple[list[str], int]:
    """(chosen source list, expected total) for progress/`__len__`."""
    known = WILDFAKE_SOURCE_COUNTS.get(config, {})
    chosen = [str(s) for s in sources] if sources is not None else list(known)
    total = 0
    for source in chosen:
        count = known.get(source)
        if count is None:
            total += max_per_source or 0  # unknown source -> can't size precisely
        elif max_per_source is not None:
            total += min(count, max_per_source)
        else:
            total += count
    return chosen, total


def eval_dataset(
    config: str = "default",
    split: str = "validation",
    *,
    sources: Sequence[str] | None = None,
    max_per_source: int | None = None,
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

    NOTE: in ``config="default"`` the real (COCO) images are all 200x200 while the
    fakes vary -- a size shortcut. Pass ``output_size=(224, 224)`` (or use
    ``config="normalized"``) for a fair evaluation.

    ::

        val = eval_dataset(output_size=(224, 224), cache_dir="wildfake_cache")
        val.warm_cache()
        detector.evaluate(val, generate_confusion_matrix=True)
    """
    chosen, total = _resolve_selection(config, sources, max_per_source)
    if total < 1:
        raise ValueError(
            f"can't size config={config!r} sources={sources!r} -- pass an explicit "
            "max_per_source= (and a known config)."
        )

    source_tag = "all" if sources is None else "+".join(sorted(chosen))
    cap_tag = f"-cap{max_per_source}" if max_per_source is not None else ""
    cache_key = f"wildfake-{config}-{split}-{source_tag}{cap_tag}-raw"

    return StreamingAugmentedDataset(
        encoded_source=lambda: iter_wildfake_encoded(
            config, split,
            sources=sources, max_per_source=max_per_source,
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
