"""Export a LightGBM binary model into a compact flat-array representation.

Format (all arrays have one entry per node, trees concatenated in depth-first
order so that the left child of node ``i`` is node ``i + 1``):

* ``values``  float64  split threshold (internal) or leaf output (leaf)
* ``right``   uint32   global index of the right child (0 for leaves)
* ``feature`` uint16   split feature index (0 for leaves)
* ``flags``   uint8    bits 0-1 missing type (0 none, 1 zero, 2 nan),
                       bit 2 default-left, bit 3 is-leaf
* ``roots``   int64    index of the first node of every tree
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

MISSING_NONE, MISSING_ZERO, MISSING_NAN = 0, 1, 2
FLAG_DEFAULT_LEFT = 0x4
FLAG_LEAF = 0x8
ZERO_THRESHOLD = 1e-35

_MISSING = {"None": MISSING_NONE, "Zero": MISSING_ZERO, "NaN": MISSING_NAN}


@dataclass
class FlatModel:
    values: np.ndarray
    right: np.ndarray
    feature: np.ndarray
    flags: np.ndarray
    roots: np.ndarray
    n_features: int
    sigmoid: float
    feature_names: list[str]

    @property
    def n_trees(self) -> int:
        return len(self.roots)

    @property
    def n_nodes(self) -> int:
        return len(self.values)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            values=self.values,
            right=self.right,
            feature=self.feature,
            flags=self.flags,
            roots=self.roots,
            n_features=np.int64(self.n_features),
            sigmoid=np.float64(self.sigmoid),
            feature_names=np.asarray(self.feature_names),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FlatModel":
        with np.load(path, allow_pickle=False) as z:
            return cls(
                values=z["values"],
                right=z["right"],
                feature=z["feature"],
                flags=z["flags"],
                roots=z["roots"],
                n_features=int(z["n_features"]),
                sigmoid=float(z["sigmoid"]),
                feature_names=[str(s) for s in z["feature_names"]],
            )


def _parse_sigmoid(objective: str) -> float:
    parts = objective.split()
    if not parts or parts[0] != "binary":
        raise ValueError(f"only binary objectives are supported, got {objective!r}")
    for p in parts[1:]:
        if p.startswith("sigmoid:"):
            return float(p.split(":", 1)[1])
    return 1.0


def flatten_booster(booster: lgb.Booster) -> FlatModel:
    dump = booster.dump_model()
    sigmoid = _parse_sigmoid(dump.get("objective", ""))
    n_features = int(dump["max_feature_idx"]) + 1
    if n_features > np.iinfo(np.uint16).max:
        raise ValueError("too many features for the flat format")

    values: list[float] = []
    right: list[int] = []
    feature: list[int] = []
    flags: list[int] = []
    roots: list[int] = []

    def emit(node: dict) -> None:
        # Iterative DFS (left before right) to keep the left child adjacent.
        stack = [node]
        while stack:
            cur = stack.pop()
            if isinstance(cur, tuple):  # ("patch", parent_index, right_node): left subtree is done
                _, parent, rnode = cur
                right[parent] = len(values)
                stack.append(rnode)
                continue
            idx = len(values)
            if "leaf_value" in cur:
                values.append(float(cur["leaf_value"]))
                right.append(0)
                feature.append(0)
                flags.append(FLAG_LEAF)
                continue
            if cur["decision_type"] != "<=":
                raise ValueError("categorical splits are not supported by the flat format")
            thr = cur["threshold"]
            if isinstance(thr, str) or not math.isfinite(float(thr)):
                raise ValueError(f"unsupported split threshold {thr!r}")
            f = _MISSING[cur["missing_type"]] | (FLAG_DEFAULT_LEFT if cur["default_left"] else 0)
            values.append(float(thr))
            right.append(0)  # patched once the left subtree has been emitted
            feature.append(int(cur["split_feature"]))
            flags.append(f)
            # Pop order: left subtree fully, then the patch marker, then right subtree.
            stack.append(("patch", idx, cur["right_child"]))
            stack.append(cur["left_child"])

    for tree in dump["tree_info"]:
        roots.append(len(values))
        emit(tree["tree_structure"])

    return FlatModel(
        values=np.asarray(values, dtype=np.float64),
        right=np.asarray(right, dtype=np.uint32),
        feature=np.asarray(feature, dtype=np.uint16),
        flags=np.asarray(flags, dtype=np.uint8),
        roots=np.asarray(roots, dtype=np.int64),
        n_features=n_features,
        sigmoid=sigmoid,
        feature_names=list(dump["feature_names"]),
    )


def reference_predict(model: FlatModel, X: np.ndarray, raw: bool = False) -> np.ndarray:
    """Slow pure-Python evaluator of the flat format (oracle for tests)."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    out = np.empty(len(X))
    for r, x in enumerate(X):
        total = 0.0
        for root in model.roots:
            i = int(root)
            while not model.flags[i] & FLAG_LEAF:
                v = x[model.feature[i]]
                fl = int(model.flags[i])
                mt = fl & 0x3
                if math.isnan(v) and mt != MISSING_NAN:
                    v = 0.0
                if (mt == MISSING_ZERO and abs(v) <= ZERO_THRESHOLD) or (mt == MISSING_NAN and math.isnan(v)):
                    left = bool(fl & FLAG_DEFAULT_LEFT)
                else:
                    left = v <= model.values[i]
                i = i + 1 if left else int(model.right[i])
            total += model.values[i]
        out[r] = total if raw else 1.0 / (1.0 + math.exp(-model.sigmoid * total))
    return out
