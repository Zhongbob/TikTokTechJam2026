"""Trainer for the fusion model's combination parameters.

Two jobs, only one of them real today:

* **weight fitting** (implemented) — ``optimal_weights()`` grid-searches the
  member-weight simplex for the split that best separates real from AI on some
  labelled example data, and reports the operating threshold to go with it. This
  is what powers ``FusionDetector(method="weighted")``.
* **meta-classifier** (stub) — a learned combiner over the member score vector.
  ``fit_meta_classifier()`` raises ``NotImplementedError`` until it's wanted.

    trainer = FusionTrainer.use_default(opensdi_repo_dir="/content/OpenSDI")
    report  = trainer.train(train_samples, objective="balanced_accuracy")
    #   -> {"weights": [0.62, 0.38], "threshold": 0.21, "balanced_accuracy": ...}
    detector = trainer.as_detector()          # FusionDetector, method="weighted"
    trainer.save("fusion_weights.json")

Members are the same two detectors `FusionDetector` uses (Community-Forensics +
OpenSDI), built via the shared `fusion.detector.build_default_members`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from shared_types.training import (
    ClassifierTrainableModel,
    LabeledImageSample,
    TrainingResult,
)

from fusion.detector import FusionDetector, build_default_members

try:  # reuse the repo's pure-python ROC-AUC
    from detector_common.base import _roc_auc as _roc_auc
except Exception:  # pragma: no cover - tiny fallback
    def _roc_auc(positives: list[float], negatives: list[float]) -> float:
        if not positives or not negatives:
            return float("nan")
        wins = ties = 0
        for p in positives:
            for n in negatives:
                if p > n:
                    wins += 1
                elif p == n:
                    ties += 1
        return (wins + 0.5 * ties) / (len(positives) * len(negatives))


_META_STUB = (
    "meta-classifier training is not implemented — the fusion currently supports "
    "the fixed 'max' rule and fitted 'weighted' rule. Use optimal_weights()/train() "
    "for the weighted method."
)

_OBJECTIVES = {"balanced_accuracy", "accuracy", "f1", "youden", "auc"}


# --- search helpers ----------------------------------------------------------

def _weight_grid(n: int, step: float) -> Iterable[tuple[float, ...]]:
    """Yield normalized weight tuples over the (n-1)-simplex on a ``step`` grid
    (e.g. n=2, step=0.02 -> (0.0,1.0), (0.02,0.98), ..., (1.0,0.0))."""
    ticks = max(1, int(round(1.0 / step)))

    def _recurse(remaining: int, slots: int) -> Iterable[tuple[int, ...]]:
        if slots == 1:
            yield (remaining,)
            return
        for i in range(remaining + 1):
            for rest in _recurse(remaining - i, slots - 1):
                yield (i, *rest)

    for combo in _recurse(ticks, n):
        yield tuple(c / ticks for c in combo)


def _fuse(row: Sequence[float], weights: Sequence[float]) -> float:
    total = sum(weights) or 1.0
    return sum(p * w for p, w in zip(row, weights)) / total


def _confusion(scores: Sequence[float], y: Sequence[int], threshold: float) -> tuple[int, int, int, int]:
    tn = fp = fn = tp = 0
    for s, label in zip(scores, y):
        pred = 1 if s >= threshold else 0
        if label == 1 and pred == 1:
            tp += 1
        elif label == 1:
            fn += 1
        elif pred == 1:
            fp += 1
        else:
            tn += 1
    return tn, fp, fn, tp


def _metrics_at(scores: Sequence[float], y: Sequence[int], threshold: float) -> dict[str, float]:
    tn, fp, fn, tp = _confusion(scores, y, threshold)
    n = tn + fp + fn + tp
    tpr = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / n if n else 0.0,
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "precision": precision,
        "recall": tpr,
        "f1": 2 * precision * tpr / (precision + tpr) if precision + tpr else 0.0,
        "fpr": fpr,
        "youden": tpr - fpr,
        "tn": float(tn), "fp": float(fp), "fn": float(fn), "tp": float(tp),
    }


def _threshold_candidates(scores: Sequence[float], step: float) -> list[float]:
    if not scores:
        return [0.5]
    lo, hi = min(scores), max(scores)
    grid = [i * step for i in range(int(1.0 / step) + 1)]
    # also try just-above every observed score so every distinct split is covered
    if len(scores) <= 2000:
        grid += [s + 1e-9 for s in scores]
    return sorted({t for t in grid if lo - step <= t <= hi + step} or {0.5})


def _pick_threshold(
    scores: Sequence[float],
    y: Sequence[int],
    *,
    objective: str,
    max_fpr: float | None,
    step: float,
) -> dict[str, float]:
    best: dict[str, float] | None = None
    best_key = -math.inf
    for t in _threshold_candidates(scores, step):
        m = _metrics_at(scores, y, t)
        if max_fpr is not None and m["fpr"] > max_fpr + 1e-9:
            continue
        key = m["recall"] if max_fpr is not None else m.get(objective, m["balanced_accuracy"])
        if key > best_key:
            best_key, best = key, m
    if best is None:  # nothing met the FPR budget — fall back to the lowest-FPR point
        best = min(
            (_metrics_at(scores, y, t) for t in _threshold_candidates(scores, step)),
            key=lambda m: (m["fpr"], -m["recall"]),
        )
    return best


# --- trainer ---------------------------------------------------------------

class FusionTrainer(ClassifierTrainableModel):
    """Fits the fusion's member weights (and its threshold) on labelled data."""

    name = "fusion-weight-trainer"

    def __init__(
        self,
        members: Sequence[Any] | None = None,
        *,
        meta_classifier: Any | None = None,
    ) -> None:
        self._members = list(members) if members is not None else []
        self._meta = meta_classifier
        #: filled by optimal_weights()/train()
        self.weights_: list[float] | None = None
        self.threshold_: float | None = None
        self.metrics_: dict[str, float] | None = None

    # --- construction ------------------------------------------------

    @classmethod
    def use_default(
        cls,
        *,
        device: str = "auto",
        opensdi_repo_dir: str | Path | None = None,
        opensdi_weights_dir: str | Path | None = None,
        opensdi_checkpoint: str | Path | None = None,
        opensdi_kwargs: dict[str, Any] | None = None,
    ) -> "FusionTrainer":
        """Build with the same two members as `FusionDetector.use_default`
        (Community-Forensics + OpenSDI)."""
        members = build_default_members(
            device=device,
            opensdi_repo_dir=opensdi_repo_dir,
            opensdi_weights_dir=opensdi_weights_dir,
            opensdi_checkpoint=opensdi_checkpoint,
            opensdi_kwargs=opensdi_kwargs,
        )
        return cls(members)

    @property
    def members(self) -> list[Any]:
        return list(self._members)

    @property
    def member_names(self) -> list[str]:
        return [getattr(m, "name", f"member{i}") for i, m in enumerate(self._members)]

    # --- feature extraction ----------------------------------------

    def member_score_matrix(
        self, samples: Iterable[LabeledImageSample]
    ) -> tuple[list[list[float]], list[int]]:
        """images -> (X: per-image member p(ai) vectors, y: binary labels).

        Runs every member once per image — the expensive step. Cache ``X, y``
        and pass them back into ``optimal_weights(X=, y=)`` to re-search cheaply.
        """
        if not self._members:
            raise RuntimeError("no members — build with FusionTrainer.use_default(...) or pass members=")

        rows: list[list[float]] = []
        labels: list[int] = []
        iterator: Iterable[LabeledImageSample] = samples
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(samples, desc="fusion: member scores", unit="img")  # type: ignore[arg-type]
        except Exception:
            pass

        for sample in iterator:
            labels.append(int(sample.metadata["binary_aigc_label"]))
            rows.append([
                float(member.predict(sample.image).ai_generated_probability)
                for member in self._members
            ])
        return rows, labels

    # --- the real job: pick weights -------------------------------

    def optimal_weights(
        self,
        samples: Iterable[LabeledImageSample] | None = None,
        *,
        objective: str = "balanced_accuracy",
        max_fpr: float | None = None,
        weight_step: float = 0.02,
        threshold_step: float = 0.005,
        top: int = 5,
        X: Sequence[Sequence[float]] | None = None,
        y: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        """Grid-search the member-weight simplex for the split that best
        separates the classes on ``samples`` (or a precomputed ``X, y``).

        objective: ``"balanced_accuracy"`` (default — robust to class imbalance),
            ``"accuracy"``, ``"f1"``, ``"youden"`` (TPR-FPR), or ``"auc"``
            (threshold-free; the reported threshold is then chosen by
            balanced accuracy).
        max_fpr: if set, only (weights, threshold) with false-positive rate
            <= this are considered, and among them recall is maximized — use
            this when you have a hard ceiling on false alarms.
        weight_step / threshold_step: grid resolution.
        top: also return the ``top`` weight vectors by objective.

        Returns a dict with ``weights`` / ``threshold`` / ``objective`` /
        ``roc_auc`` and the full metric set at that operating point, plus
        ``ranked`` (list of the best ``top``). Also stores ``weights_`` /
        ``threshold_`` / ``metrics_`` on the trainer.
        """
        if objective not in _OBJECTIVES:
            raise ValueError(f"objective must be one of {sorted(_OBJECTIVES)}")
        if X is None or y is None:
            if samples is None:
                raise ValueError("pass either samples= or both X= and y=")
            X, y = self.member_score_matrix(samples)
        X = [list(map(float, row)) for row in X]
        y = [int(v) for v in y]
        if not X:
            raise ValueError("no samples")
        n_members = len(X[0])
        if not any(t == 1 for t in y) or not any(t == 0 for t in y):
            raise ValueError("need both classes (0 and 1) present in the labels")

        scored: list[tuple[float, tuple[float, ...], dict[str, float]]] = []
        for weights in _weight_grid(n_members, weight_step):
            fused = [_fuse(row, weights) for row in X]
            if objective == "auc" and max_fpr is None:
                pos = [s for s, t in zip(fused, y) if t == 1]
                neg = [s for s, t in zip(fused, y) if t == 0]
                key = _roc_auc(pos, neg)
                op = _pick_threshold(fused, y, objective="balanced_accuracy",
                                     max_fpr=None, step=threshold_step)
            else:
                op = _pick_threshold(fused, y, objective=objective,
                                     max_fpr=max_fpr, step=threshold_step)
                key = op["recall"] if max_fpr is not None else op[objective]
            scored.append((key, weights, op))

        scored.sort(key=lambda item: item[0], reverse=True)
        best_key, best_weights, best_op = scored[0]

        fused_best = [_fuse(row, best_weights) for row in X]
        pos = [s for s, t in zip(fused_best, y) if t == 1]
        neg = [s for s, t in zip(fused_best, y) if t == 0]

        result: dict[str, Any] = {
            "weights": [round(w, 4) for w in best_weights],
            "threshold": round(best_op["threshold"], 4),
            "objective": objective,
            "objective_value": round(best_key, 4),
            "roc_auc": round(_roc_auc(pos, neg), 4),
            "n_samples": len(y),
            "member_names": self.member_names or [f"member{i}" for i in range(n_members)],
            **{k: round(v, 4) for k, v in best_op.items() if k != "threshold"},
            "ranked": [
                {"weights": [round(w, 4) for w in wts],
                 "threshold": round(op["threshold"], 4),
                 objective: round(k, 4),
                 "fpr": round(op["fpr"], 4),
                 "recall": round(op["recall"], 4)}
                for k, wts, op in scored[:top]
            ],
        }
        self.weights_ = list(result["weights"])
        self.threshold_ = float(result["threshold"])
        self.metrics_ = {k: v for k, v in result.items()
                         if isinstance(v, (int, float))}
        return result

    # --- TrainableModel surface ----------------------------------

    def train(
        self, samples: Iterable[LabeledImageSample], **kwargs: Any
    ) -> TrainingResult:
        """Fit the member weights + threshold (thin wrapper over
        ``optimal_weights``). ``kwargs`` are forwarded to it."""
        report = self.optimal_weights(samples, **kwargs)
        return TrainingResult(
            epochs_completed=1,
            final_loss=None,
            metrics={k: float(v) for k, v in report.items() if isinstance(v, (int, float))},
            notes=(f"weights={report['weights']} threshold={report['threshold']} "
                   f"({report['objective']}={report['objective_value']}, "
                   f"roc_auc={report['roc_auc']})"),
        )

    def evaluate(
        self, samples: Iterable[LabeledImageSample], **kwargs: Any
    ) -> dict[str, float]:
        """Score the *current* fitted weights/threshold on held-out ``samples``."""
        if self.weights_ is None or self.threshold_ is None:
            raise RuntimeError("nothing fitted yet — call train()/optimal_weights() first")
        X, y = self.member_score_matrix(samples)
        fused = [_fuse(row, self.weights_) for row in X]
        pos = [s for s, t in zip(fused, y) if t == 1]
        neg = [s for s, t in zip(fused, y) if t == 0]
        metrics = _metrics_at(fused, y, self.threshold_)
        metrics["roc_auc"] = _roc_auc(pos, neg)
        metrics["n_samples"] = float(len(y))
        return {k: float(v) for k, v in metrics.items()}

    def save(self, path: str | Path) -> None:
        """Persist the fitted weights + threshold (JSON). Members are not saved —
        they come from `FusionTrainer.use_default` / your own list."""
        if self.weights_ is None:
            raise RuntimeError("nothing fitted yet — call train()/optimal_weights() first")
        Path(path).write_text(json.dumps({
            "weights": self.weights_,
            "threshold": self.threshold_,
            "member_names": self.member_names,
            "metrics": self.metrics_ or {},
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, *, members: Sequence[Any] | None = None) -> "FusionTrainer":
        """Reload fitted weights from `save()`. Pass ``members=`` (or call
        ``attach_members`` after) to make ``as_detector()`` usable."""
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        trainer = cls(members)
        trainer.weights_ = [float(w) for w in blob["weights"]]
        trainer.threshold_ = float(blob["threshold"])
        trainer.metrics_ = blob.get("metrics") or None
        return trainer

    # --- handoff to inference ------------------------------------

    def attach_members(self, members: Sequence[Any]) -> None:
        self._members = list(members)

    def as_detector(
        self,
        *,
        members: Sequence[Any] | None = None,
        decision_threshold: float | None = None,
        **fusion_kwargs: Any,
    ) -> FusionDetector:
        """Build a ``method="weighted"`` `FusionDetector` from the fitted weights."""
        if self.weights_ is None:
            raise RuntimeError("nothing fitted yet — call train()/optimal_weights() first")
        use_members = list(members) if members is not None else self._members
        if not use_members:
            raise RuntimeError("no members — pass members= or build via FusionTrainer.use_default")
        return FusionDetector(
            use_members,
            method="weighted",
            weights=self.weights_,
            decision_threshold=(self.threshold_ if decision_threshold is None
                                else float(decision_threshold)),
            **fusion_kwargs,
        )

    # --- meta-classifier (future) -------------------------------

    def fit_meta_classifier(self, samples: Iterable[LabeledImageSample], **kwargs: Any) -> Any:
        """Train a learned combiner over the member score vectors. **Not implemented.**"""
        raise NotImplementedError(_META_STUB)
