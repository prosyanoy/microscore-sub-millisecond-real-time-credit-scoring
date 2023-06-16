"""Interchangeable scoring engines: LightGBM (Python), ONNX Runtime, custom C++."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import lightgbm as lgb
import numpy as np

from .flatten import FlatModel


class Engine(Protocol):
    name: str

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (n, n_features) float64 -> default probabilities (n,)."""

    def predict_row(self, x: np.ndarray) -> float:
        """x: (n_features,) float64 -> default probability."""


class LightGBMEngine:
    def __init__(self, booster: lgb.Booster, num_threads: int | None = None):
        self.name = "lightgbm" if num_threads is None else f"lightgbm-{num_threads}t"
        self._booster = booster
        self._kw = {} if num_threads is None else {"num_threads": num_threads}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._booster.predict(X, **self._kw))

    def predict_row(self, x: np.ndarray) -> float:
        return float(self._booster.predict(x.reshape(1, -1), **self._kw)[0])


class OnnxEngine:
    name = "onnxruntime"

    def __init__(self, onnx_path: str | Path, num_threads: int = 1):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
        self._input = self._sess.get_inputs()[0].name
        # Output 1 is the per-class probability matrix (N, 2); we need column 1.
        self._outputs = [self._sess.get_outputs()[1].name]

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self._sess.run(self._outputs, {self._input: X.astype(np.float32, copy=False)})[0]
        return np.asarray(probs)[:, 1].astype(np.float64)

    def predict_row(self, x: np.ndarray) -> float:
        return float(self.predict(x.reshape(1, -1))[0])


class CppEngine:
    def __init__(self, flat: FlatModel, n_threads: int = 1):
        from . import _treeeval

        self.name = "cpp" if n_threads == 1 else f"cpp-{n_threads}t"
        self.n_threads = n_threads
        self._ens = _treeeval.FlatEnsemble(
            flat.values, flat.right, flat.feature, flat.flags, flat.roots, flat.n_features, flat.sigmoid
        )

    @property
    def ensemble(self):
        return self._ens

    def predict(self, X: np.ndarray, raw: bool = False) -> np.ndarray:
        return self._ens.predict_batch(X, raw, self.n_threads)

    def predict_row(self, x: np.ndarray) -> float:
        return self._ens.predict_one(x, False)
