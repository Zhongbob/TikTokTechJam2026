"""Tree-based meta-classifier for the fusion model.

A linear blend of two *complementary specialists* dilutes whichever member is
carrying the signal. A tree instead learns threshold-branching logic —
"if synthetic_prob > 0.6 → fake; else if tamper_score > 0.3 → fake; else look
at both together" — which is the natural generalization of the OR-rule and can
also pick up interaction effects (moderate-on-both being suspicious even when
neither crosses its own threshold).

`MetaClassifier` wraps an estimator + a feature spec so that at inference it
takes the *raw member p(ai) vector* and applies the same feature transform the
fit used. It works with:

* the built-in dependency-free `_SimpleTree` (``kind="tree"``, the default),
* ``sklearn`` (``kind="sklearn-tree" | "forest" | "gboost"``),
* ``xgboost`` (``kind="xgboost"``),
* any estimator you pass as ``estimator=`` with ``fit`` / ``predict_proba``.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Sequence

# feature_spec -> extra columns appended after the raw member probabilities
FEATURE_SPECS: dict[str, list[str]] = {
    "probs": [],
    "augmented": ["max", "min", "mean", "abs_diff", "product"],
}

_EXTRA_FUNCS = {
    "max": lambda r: max(r),
    "min": lambda r: min(r),
    "mean": lambda r: sum(r) / len(r),
    "abs_diff": lambda r: abs(r[0] - r[1]) if len(r) >= 2 else 0.0,
    "product": lambda r: math.prod(r),
    "sum": lambda r: sum(r),
}


def _extras_for(feature_spec: str | Sequence[str]) -> list[str]:
    if isinstance(feature_spec, str):
        if feature_spec not in FEATURE_SPECS:
            raise ValueError(f"feature_spec must be one of {sorted(FEATURE_SPECS)} or a list of "
                             f"{sorted(_EXTRA_FUNCS)}")
        return list(FEATURE_SPECS[feature_spec])
    return list(feature_spec)


def build_feature_matrix(
    rows: Sequence[Sequence[float]], feature_spec: str | Sequence[str] = "probs"
) -> list[list[float]]:
    """Raw member-prob rows -> feature rows (raw probs + any extras)."""
    extras = _extras_for(feature_spec)
    out: list[list[float]] = []
    for row in rows:
        row = [float(v) for v in row]
        out.append(row + [float(_EXTRA_FUNCS[name](row)) for name in extras])
    return out


# --- dependency-free shallow CART ------------------------------------------

class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "proba")

    def __init__(self) -> None:
        self.feature: int | None = None
        self.threshold: float | None = None
        self.left: _Node | None = None
        self.right: _Node | None = None
        self.proba: float = 0.0  # P(class == 1) at a leaf


class _SimpleTree:
    """Tiny binary CART (Gini), enough to combine a few detector scores.

    ``class_weight="balanced"`` reweights so the minority class isn't ignored —
    matches ``sklearn``'s option and matters for imbalanced eval sets.
    """

    def __init__(
        self,
        *,
        max_depth: int = 3,
        min_samples_leaf: int = 10,
        min_samples_split: int = 20,
        max_thresholds: int = 64,
        class_weight: str | dict[int, float] | None = "balanced",
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_thresholds = max_thresholds
        self.class_weight = class_weight
        self.classes_ = [0, 1]
        self.root_: _Node | None = None
        self.feature_importances_: list[float] = []
        self.n_features_: int = 0

    # -- fit --
    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> "_SimpleTree":
        rows = [[float(v) for v in r] for r in X]
        labels = [int(v) for v in y]
        self.n_features_ = len(rows[0]) if rows else 0
        n_pos = sum(labels) or 1
        n_neg = (len(labels) - sum(labels)) or 1
        if self.class_weight == "balanced":
            w = {1: len(labels) / (2 * n_pos), 0: len(labels) / (2 * n_neg)}
        elif isinstance(self.class_weight, dict):
            w = self.class_weight
        else:
            w = {0: 1.0, 1: 1.0}
        self._w = w
        self._importance = [0.0] * self.n_features_
        self.root_ = self._build(rows, labels, depth=0)
        total = sum(self._importance) or 1.0
        self.feature_importances_ = [v / total for v in self._importance]
        return self

    def _wcounts(self, labels: Sequence[int]) -> tuple[float, float]:
        wp = sum(self._w[1] for v in labels if v == 1)
        wn = sum(self._w[0] for v in labels if v == 0)
        return wp, wn

    @staticmethod
    def _gini(wp: float, wn: float) -> float:
        tot = wp + wn
        if tot <= 0:
            return 0.0
        p = wp / tot
        return 1.0 - p * p - (1.0 - p) ** 2

    def _build(self, rows: list[list[float]], labels: list[int], depth: int) -> _Node:
        node = _Node()
        wp, wn = self._wcounts(labels)
        node.proba = wp / (wp + wn) if (wp + wn) else 0.0

        if (depth >= self.max_depth or len(labels) < self.min_samples_split
                or wp == 0 or wn == 0):
            return node

        parent_gini = self._gini(wp, wn)
        parent_w = wp + wn
        best = (0.0, None, None)  # (weighted impurity decrease, feature, threshold)

        for f in range(self.n_features_):
            values = sorted({r[f] for r in rows})
            if len(values) < 2:
                continue
            cands = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
            if len(cands) > self.max_thresholds:
                step = len(cands) / self.max_thresholds
                cands = [cands[int(i * step)] for i in range(self.max_thresholds)]
            for thr in cands:
                left = [lab for r, lab in zip(rows, labels) if r[f] <= thr]
                right = [lab for r, lab in zip(rows, labels) if r[f] > thr]
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue
                lwp, lwn = self._wcounts(left)
                rwp, rwn = self._wcounts(right)
                child = ((lwp + lwn) * self._gini(lwp, lwn)
                         + (rwp + rwn) * self._gini(rwp, rwn)) / parent_w
                decrease = parent_gini - child
                if decrease > best[0]:
                    best = (decrease, f, thr)

        if best[1] is None:
            return node

        decrease, f, thr = best
        self._importance[f] += decrease * parent_w
        node.feature, node.threshold = f, thr
        lrows = [r for r in rows if r[f] <= thr]
        llabs = [lab for r, lab in zip(rows, labels) if r[f] <= thr]
        rrows = [r for r in rows if r[f] > thr]
        rlabs = [lab for r, lab in zip(rows, labels) if r[f] > thr]
        node.left = self._build(lrows, llabs, depth + 1)
        node.right = self._build(rrows, rlabs, depth + 1)
        return node

    # -- predict --
    def _leaf_proba(self, row: Sequence[float]) -> float:
        node = self.root_
        assert node is not None
        while node.feature is not None:
            node = node.left if row[node.feature] <= node.threshold else node.right  # type: ignore[assignment]
        return node.proba

    def predict_proba(self, X: Sequence[Sequence[float]]) -> list[list[float]]:
        out = []
        for row in X:
            p1 = self._leaf_proba([float(v) for v in row])
            out.append([1.0 - p1, p1])
        return out

    def predict(self, X: Sequence[Sequence[float]]) -> list[int]:
        return [1 if p[1] >= 0.5 else 0 for p in self.predict_proba(X)]

    def rules_text(self, feature_names: Sequence[str] | None = None) -> str:
        names = list(feature_names) if feature_names else [f"f{i}" for i in range(self.n_features_)]
        lines: list[str] = []

        def walk(node: _Node, depth: int) -> None:
            pad = "  " * depth
            if node.feature is None:
                lines.append(f"{pad}-> p(fake)={node.proba:.3f}")
                return
            name = names[node.feature] if node.feature < len(names) else f"f{node.feature}"
            lines.append(f"{pad}if {name} <= {node.threshold:.4f}:")
            walk(node.left, depth + 1)  # type: ignore[arg-type]
            lines.append(f"{pad}else:  # {name} > {node.threshold:.4f}")
            walk(node.right, depth + 1)  # type: ignore[arg-type]

        if self.root_ is not None:
            walk(self.root_, 0)
        return "\n".join(lines)


# --- estimator factory ---------------------------------------------------

def default_estimator(kind: str = "tree", **kwargs: Any) -> Any:
    """``kind`` -> an unfitted estimator with ``fit`` / ``predict_proba``."""
    kind = kind.lower()
    if kind == "tree":
        params = dict(max_depth=3, min_samples_leaf=10, class_weight="balanced")
        params.update(kwargs)
        return _SimpleTree(**params)
    if kind in ("sklearn-tree", "dtree", "decisiontree"):
        from sklearn.tree import DecisionTreeClassifier

        params = dict(max_depth=3, min_samples_leaf=10, class_weight="balanced")
        params.update(kwargs)
        return DecisionTreeClassifier(**params)
    if kind in ("forest", "rf", "randomforest"):
        from sklearn.ensemble import RandomForestClassifier

        params = dict(n_estimators=200, max_depth=4, min_samples_leaf=5,
                      class_weight="balanced_subsample")
        params.update(kwargs)
        return RandomForestClassifier(**params)
    if kind in ("gboost", "gb", "hgb", "boost"):
        from sklearn.ensemble import HistGradientBoostingClassifier

        params = dict(max_depth=3, max_iter=200, learning_rate=0.1)
        params.update(kwargs)
        return HistGradientBoostingClassifier(**params)
    if kind in ("xgboost", "xgb"):
        from xgboost import XGBClassifier

        params = dict(n_estimators=200, max_depth=3, learning_rate=0.1,
                      subsample=0.9, colsample_bytree=0.9, eval_metric="logloss")
        params.update(kwargs)
        return XGBClassifier(**params)
    raise ValueError(
        "kind must be one of: tree, sklearn-tree, forest, gboost, xgboost "
        f"(got {kind!r})"
    )


# --- the wrapper the fusion detector holds -----------------------------

class MetaClassifier:
    """Estimator + feature spec. Call `predict_fake_proba(rows)` with raw member
    p(ai) vectors — the feature transform is applied internally."""

    def __init__(
        self,
        estimator: Any,
        *,
        feature_spec: str | Sequence[str] = "probs",
        member_names: Sequence[str] | None = None,
    ) -> None:
        self.estimator = estimator
        self.feature_spec = feature_spec
        self.member_names = list(member_names) if member_names else None
        self.n_members = len(self.member_names) if self.member_names else 0
        self._pos_col = 1

    # -- fit / predict --
    def fit(self, rows: Sequence[Sequence[float]], y: Sequence[int]) -> "MetaClassifier":
        if rows:
            self.n_members = len(rows[0])
        X = build_feature_matrix(rows, self.feature_spec)
        self.estimator.fit(X, list(int(v) for v in y))
        classes = list(getattr(self.estimator, "classes_", [0, 1]))
        self._pos_col = classes.index(1) if 1 in classes else len(classes) - 1
        return self

    def predict_fake_proba(self, rows: Sequence[Sequence[float]]) -> list[float]:
        X = build_feature_matrix(rows, self.feature_spec)
        proba = self.estimator.predict_proba(X)
        return [float(p[self._pos_col]) for p in proba]

    def predict(self, rows: Sequence[Sequence[float]], threshold: float = 0.5) -> list[int]:
        return [1 if p >= threshold else 0 for p in self.predict_fake_proba(rows)]

    # -- introspection --
    @property
    def feature_names(self) -> list[str]:
        base = self.member_names or [f"member{i}" for i in range(self.n_members)]
        return list(base) + _extras_for(self.feature_spec)

    @property
    def feature_importances_(self) -> dict[str, float] | None:
        imp = getattr(self.estimator, "feature_importances_", None)
        if imp is None:
            return None
        return {name: float(v) for name, v in zip(self.feature_names, imp)}

    def rules_text(self) -> str:
        if isinstance(self.estimator, _SimpleTree):
            return self.estimator.rules_text(self.feature_names)
        try:
            from sklearn.tree import export_text

            return export_text(self.estimator, feature_names=list(self.feature_names))
        except Exception:
            return ""

    # -- persistence --
    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self))

    @classmethod
    def load(cls, path: str | Path) -> "MetaClassifier":
        obj = pickle.loads(Path(path).read_bytes())
        if not isinstance(obj, cls):
            raise TypeError(f"{path} is not a pickled MetaClassifier")
        return obj
