"""Trainer for the fusion model's combination parameters.

Fits the parameters for two of `FusionDetector`'s methods:

* **``method="weighted"``** — ``optimal_weights()`` grid-searches the
  member-weight simplex for the linear split that best separates real from AI,
  plus its operating threshold.
* **``method="meta"``** — ``fit_meta_classifier()`` trains a small **tree-based**
  combiner (dependency-free CART by default, or ``sklearn`` / ``xgboost``) over
  the member score vector. Trees learn threshold-branching logic, which fits two
  non-corroborating specialists far better than a linear blend.

``compare_methods()`` fits both and evaluates ``max`` / ``weighted`` / ``meta``
side by side on a held-out split so you can pick the winner.

    trainer = FusionTrainer.use_default(opensdi_repo_dir="/content/OpenSDI")
    table   = trainer.compare_methods(train_samples, val_samples)
    detector = trainer.as_detector(method="meta")     # or "weighted" / "max"

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


_OBJECTIVES = {"balanced_accuracy", "accuracy", "f1", "youden", "auc"}


def _classify_report(scores: Sequence[float], y: Sequence[int], threshold: float) -> dict[str, float]:
    """Full metric set for a score vector at one threshold, plus ROC-AUC."""
    pos = [s for s, t in zip(scores, y) if t == 1]
    neg = [s for s, t in zip(scores, y) if t == 0]
    metrics = _metrics_at(scores, y, threshold)
    metrics["roc_auc"] = _roc_auc(pos, neg)
    metrics["n_samples"] = float(len(y))
    return {k: (round(float(v), 4) if isinstance(v, float) else float(v)) for k, v in metrics.items()}


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
        #: filled by optimal_weights()/train()
        self.weights_: list[float] | None = None
        self.threshold_: float | None = None
        self.metrics_: dict[str, float] | None = None
        #: filled by fit_meta_classifier()
        self.meta_: Any | None = meta_classifier
        self.meta_threshold_: float = 0.5

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
        self,
        samples: Iterable[LabeledImageSample],
        *,
        method: str | None = None,
        decision_threshold: float | None = None,
        X: Sequence[Sequence[float]] | None = None,
        y: Sequence[int] | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Score one fusion method on ``samples`` (or a precomputed ``X, y``).

        ``method``: ``"max"`` (no fitting needed — the default OR-rule at
        ``DEFAULT_MAX_THRESHOLD``), ``"mean"``, ``"weighted"`` (needs
        ``optimal_weights()``/``train()``), or ``"meta"`` (needs
        ``fit_meta_classifier()``). Omit it to use whatever is fitted
        (meta > weighted), falling back to ``"max"``.
        """
        if X is None or y is None:
            X, y = self.member_score_matrix(samples)
        X = [list(map(float, r)) for r in X]
        y = [int(v) for v in y]

        if method is None:
            method = ("meta" if self.meta_ is not None
                      else "weighted" if self.weights_ is not None
                      else "max")

        if method == "meta":
            if self.meta_ is None:
                raise RuntimeError("method='meta' but no meta classifier — call fit_meta_classifier() first")
            thr = self.meta_threshold_ if decision_threshold is None else float(decision_threshold)
            return _classify_report(self.meta_.predict_fake_proba(X), y, thr)
        if method == "weighted":
            if self.weights_ is None or self.threshold_ is None:
                raise RuntimeError("method='weighted' but nothing fitted — call optimal_weights()/train() first")
            thr = self.threshold_ if decision_threshold is None else float(decision_threshold)
            return _classify_report([_fuse(r, self.weights_) for r in X], y, thr)
        if method in ("max", "mean"):
            from fusion.detector import DEFAULT_MAX_THRESHOLD

            fused = [max(r) for r in X] if method == "max" else [sum(r) / len(r) for r in X]
            default_thr = DEFAULT_MAX_THRESHOLD if method == "max" else 0.5
            thr = default_thr if decision_threshold is None else float(decision_threshold)
            return _classify_report(fused, y, thr)
        raise ValueError("method must be 'max', 'mean', 'weighted', or 'meta'")

    # --- the other real job: fit a tree-based meta-classifier -----

    def fit_meta_classifier(
        self,
        samples: Iterable[LabeledImageSample] | None = None,
        *,
        kind: str = "tree",
        feature_spec: str | Sequence[str] = "probs",
        estimator: Any | None = None,
        objective: str = "balanced_accuracy",
        max_fpr: float | None = None,
        tune_threshold: bool = True,
        threshold_step: float = 0.005,
        X: Sequence[Sequence[float]] | None = None,
        y: Sequence[int] | None = None,
        val_samples: Iterable[LabeledImageSample] | None = None,
        X_val: Sequence[Sequence[float]] | None = None,
        y_val: Sequence[int] | None = None,
        **estimator_kwargs: Any,
    ) -> dict[str, Any]:
        """Train a tree-based combiner over the member score vectors.

        kind: ``"tree"`` (default, dependency-free shallow CART), ``"sklearn-tree"``,
            ``"forest"``, ``"gboost"`` (HistGradientBoosting), or ``"xgboost"``.
            Ignored if ``estimator=`` (any ``fit`` / ``predict_proba`` object) is given.
        feature_spec: ``"probs"`` (default — just the raw member p(ai)) or
            ``"augmented"`` (also max / min / mean / abs_diff / product), or a
            custom list of those names.
        tune_threshold: sweep the meta output for the best ``objective`` (or best
            recall subject to ``max_fpr``); otherwise the threshold stays 0.5.

        Fits ``self.meta_`` / ``self.meta_threshold_`` and returns a report with
        ``feature_importances``, the tree ``rules`` (for tree kinds), and
        ``train`` (+ ``val`` if val data given) metric dicts.
        """
        from fusion._meta import MetaClassifier, default_estimator

        if X is None or y is None:
            if samples is None:
                raise ValueError("pass either samples= or both X= and y=")
            X, y = self.member_score_matrix(samples)
        X = [list(map(float, row)) for row in X]
        y = [int(v) for v in y]
        if not any(t == 1 for t in y) or not any(t == 0 for t in y):
            raise ValueError("need both classes (0 and 1) present in the labels")

        est = estimator if estimator is not None else default_estimator(kind, **estimator_kwargs)
        meta = MetaClassifier(est, feature_spec=feature_spec, member_names=self.member_names)
        meta.fit(X, y)

        train_scores = meta.predict_fake_proba(X)
        if tune_threshold:
            op = _pick_threshold(train_scores, y, objective=objective,
                                 max_fpr=max_fpr, step=threshold_step)
            meta_threshold = op["threshold"]
        else:
            meta_threshold = 0.5

        self.meta_ = meta
        self.meta_threshold_ = float(meta_threshold)

        report: dict[str, Any] = {
            "kind": "custom" if estimator is not None else kind,
            "feature_spec": feature_spec,
            "feature_names": meta.feature_names,
            "feature_importances": meta.feature_importances_,
            "threshold": round(self.meta_threshold_, 4),
            "objective": objective,
            "n_samples": len(y),
            "member_names": self.member_names,
            "train": _classify_report(train_scores, y, self.meta_threshold_),
        }
        rules = meta.rules_text()
        if rules:
            report["rules"] = rules

        if X_val is None and val_samples is not None:
            X_val, y_val = self.member_score_matrix(val_samples)
        if X_val is not None and y_val is not None:
            X_val = [list(map(float, r)) for r in X_val]
            y_val = [int(v) for v in y_val]
            report["val"] = _classify_report(meta.predict_fake_proba(X_val), y_val, self.meta_threshold_)

        return report

    # --- experiment harness: compare every method ---------------

    def compare_methods(
        self,
        train_samples: Iterable[LabeledImageSample] | None = None,
        val_samples: Iterable[LabeledImageSample] | None = None,
        *,
        X_train: Sequence[Sequence[float]] | None = None,
        y_train: Sequence[int] | None = None,
        X_val: Sequence[Sequence[float]] | None = None,
        y_val: Sequence[int] | None = None,
        objective: str = "balanced_accuracy",
        max_fpr: float | None = None,
        meta_kinds: Sequence[str] = ("tree", "gboost"),
        feature_specs: Sequence[str] = ("probs",),
        fixed_max_threshold: float | None = None,
    ) -> dict[str, dict[str, float]]:
        """Fit ``weighted`` + every ``meta`` variant on the train split and score
        ``max`` / ``weighted`` / ``meta`` (and each bare member) on the val split
        (falls back to the train split, with a warning, if no val given).

        Returns ``{method_name: metric_dict}`` — sort by ``balanced_accuracy`` or
        whatever you care about. ``meta_kinds`` you don't have installed are
        skipped with a note.

        This is a read-only benchmark: it restores the trainer's fitted state
        afterwards, so run ``optimal_weights()`` / ``fit_meta_classifier()`` for
        the method you pick.
        """
        from fusion.detector import DEFAULT_MAX_THRESHOLD

        snapshot = (self.weights_, self.threshold_, self.metrics_,
                    self.meta_, self.meta_threshold_)

        if X_train is None or y_train is None:
            if train_samples is None:
                raise ValueError("pass train_samples= or X_train=/y_train=")
            X_train, y_train = self.member_score_matrix(train_samples)
        X_train = [list(map(float, r)) for r in X_train]
        y_train = [int(v) for v in y_train]

        if X_val is None or y_val is None:
            if val_samples is not None:
                X_val, y_val = self.member_score_matrix(val_samples)
            else:
                print("[fusion] compare_methods: no val split — scoring on the train "
                      "split (optimistic).")
                X_val, y_val = X_train, y_train
        X_val = [list(map(float, r)) for r in X_val]
        y_val = [int(v) for v in y_val]

        results: dict[str, dict[str, float]] = {}

        # bare members
        for i, name in enumerate(self.member_names or [f"member{i}" for i in range(len(X_train[0]))]):
            col_tr = [r[i] for r in X_train]
            col_va = [r[i] for r in X_val]
            op = _pick_threshold(col_tr, y_train, objective=objective, max_fpr=max_fpr, step=0.005)
            results[f"member:{name}"] = _classify_report(col_va, y_val, op["threshold"])

        # max
        max_tr = [max(r) for r in X_train]
        max_va = [max(r) for r in X_val]
        thr = (fixed_max_threshold if fixed_max_threshold is not None
               else _pick_threshold(max_tr, y_train, objective=objective,
                                    max_fpr=max_fpr, step=0.005)["threshold"])
        results["max"] = _classify_report(max_va, y_val, thr)
        results["max@default"] = _classify_report(max_va, y_val, DEFAULT_MAX_THRESHOLD)

        # weighted
        self.optimal_weights(X=X_train, y=y_train, objective=objective, max_fpr=max_fpr)
        w_va = [_fuse(r, self.weights_) for r in X_val]
        results["weighted"] = _classify_report(w_va, y_val, self.threshold_)

        # meta variants
        for kind in meta_kinds:
            for spec in feature_specs:
                tag = f"meta:{kind}" + ("" if spec == "probs" else f"[{spec}]")
                try:
                    rep = self.fit_meta_classifier(
                        X=X_train, y=y_train, kind=kind, feature_spec=spec,
                        objective=objective, max_fpr=max_fpr,
                        X_val=X_val, y_val=y_val,
                    )
                except ImportError as error:
                    print(f"[fusion] skipping {tag}: {error}")
                    continue
                results[tag] = rep["val"]

        (self.weights_, self.threshold_, self.metrics_,
         self.meta_, self.meta_threshold_) = snapshot
        return results

    # --- persistence -------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the fitted parameters. Writes a JSON manifest at ``path``; if a
        meta classifier is fitted it also writes ``<path>.meta.pkl`` alongside.
        Members are not saved (rebuild via `FusionTrainer.use_default`)."""
        path = Path(path)
        manifest: dict[str, Any] = {
            "member_names": self.member_names,
            "weights": self.weights_,
            "threshold": self.threshold_,
            "metrics": self.metrics_ or {},
            "meta_threshold": self.meta_threshold_,
            "has_meta": self.meta_ is not None,
        }
        if self.meta_ is not None:
            meta_path = path.with_suffix(path.suffix + ".meta.pkl")
            self.meta_.save(meta_path)
            manifest["meta_file"] = meta_path.name
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, *, members: Sequence[Any] | None = None) -> "FusionTrainer":
        """Reload a `save()` bundle. Pass ``members=`` to make ``as_detector()`` usable."""
        path = Path(path)
        blob = json.loads(path.read_text(encoding="utf-8"))
        trainer = cls(members)
        if blob.get("weights") is not None:
            trainer.weights_ = [float(w) for w in blob["weights"]]
        trainer.threshold_ = blob.get("threshold")
        trainer.metrics_ = blob.get("metrics") or None
        trainer.meta_threshold_ = float(blob.get("meta_threshold", 0.5))
        if blob.get("has_meta"):
            from fusion._meta import MetaClassifier

            meta_path = path.with_name(blob.get("meta_file", path.name + ".meta.pkl"))
            trainer.meta_ = MetaClassifier.load(meta_path)
        return trainer

    # --- handoff to inference ------------------------------------

    def attach_members(self, members: Sequence[Any]) -> None:
        self._members = list(members)

    def as_detector(
        self,
        *,
        method: str = "meta",
        members: Sequence[Any] | None = None,
        decision_threshold: float | None = None,
        **fusion_kwargs: Any,
    ) -> FusionDetector:
        """Build a `FusionDetector` from what's been fitted.

        ``method``: ``"meta"`` (default, needs ``fit_meta_classifier``),
        ``"weighted"`` (needs ``optimal_weights``/``train``), or ``"max"``.
        """
        use_members = list(members) if members is not None else self._members
        if not use_members:
            raise RuntimeError("no members — pass members= or build via FusionTrainer.use_default")

        if method == "meta":
            if self.meta_ is None:
                raise RuntimeError("no meta classifier — call fit_meta_classifier() first")
            det = FusionDetector(
                use_members, method="max",  # placeholder; replaced below
                decision_threshold=(self.meta_threshold_ if decision_threshold is None
                                    else float(decision_threshold)),
                **fusion_kwargs,
            )
            det.attach_meta_classifier(self.meta_)
            if decision_threshold is None:
                det.decision_threshold = self.meta_threshold_
            return det
        if method == "weighted":
            if self.weights_ is None:
                raise RuntimeError("no weights — call optimal_weights()/train() first")
            return FusionDetector(
                use_members, method="weighted", weights=self.weights_,
                decision_threshold=(self.threshold_ if decision_threshold is None
                                    else float(decision_threshold)),
                **fusion_kwargs,
            )
        if method == "max":
            return FusionDetector(
                use_members, method="max",
                decision_threshold=decision_threshold, **fusion_kwargs,
            )
        raise ValueError("method must be 'meta', 'weighted', or 'max'")
