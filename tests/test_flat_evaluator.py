import math

import lightgbm as lgb
import numpy as np
import pytest

from microscore import _treeeval
from microscore.engines import CppEngine
from microscore.flatten import FlatModel, flatten_booster, reference_predict

TOL = 1e-6


@pytest.fixture(scope="module")
def flat(booster):
    return flatten_booster(booster)


@pytest.fixture(scope="module")
def cpp(flat):
    return CppEngine(flat)


def test_batch_matches_lightgbm(booster, cpp, X):
    assert np.abs(cpp.predict(X) - booster.predict(X)).max() < TOL


def test_single_row_matches_lightgbm(booster, cpp, X):
    expected = booster.predict(X[:200])
    got = np.array([cpp.predict_row(r) for r in X[:200]])
    assert np.abs(got - expected).max() < TOL


def test_raw_scores_match(booster, flat, X):
    e = CppEngine(flat)
    assert np.abs(e.predict(X, raw=True) - booster.predict(X, raw_score=True)).max() < TOL


@pytest.mark.parametrize("threads", [2, 4, 0])
def test_multithreaded_equals_single_thread(flat, X, threads):
    big = np.tile(X, (3, 1))  # enough rows for several workers
    assert np.array_equal(CppEngine(flat, threads).predict(big), CppEngine(flat, 1).predict(big))


def test_missing_and_zero_values(booster, cpp, X):
    Xm = X[:500].copy()
    rng = np.random.default_rng(0)
    Xm[rng.random(Xm.shape) < 0.2] = np.nan
    Xm[rng.random(Xm.shape) < 0.1] = 0.0
    assert np.abs(cpp.predict(Xm) - booster.predict(Xm)).max() < TOL


def test_reference_python_evaluator_agrees(booster, flat, X):
    ref = reference_predict(flat, X[:100])
    assert np.abs(ref - booster.predict(X[:100])).max() < TOL


def test_save_load_roundtrip(flat, X, tmp_path):
    flat.save(tmp_path / "m.npz")
    again = FlatModel.load(tmp_path / "m.npz")
    assert np.array_equal(CppEngine(again).predict(X[:50]), CppEngine(flat).predict(X[:50]))
    assert again.feature_names == flat.feature_names


def test_node_layout_is_compact(flat):
    # 8 + 4 + 2 + 1 bytes of payload per node, padded to 16 in the C++ struct.
    payload = flat.values.nbytes + flat.right.nbytes + flat.feature.nbytes + flat.flags.nbytes
    assert payload == 15 * flat.n_nodes
    assert flat.n_trees == 40


def test_shape_validation(cpp):
    with pytest.raises(ValueError):
        cpp.ensemble.predict_one(np.zeros(3))
    with pytest.raises(ValueError):
        cpp.ensemble.predict_batch(np.zeros((2, 3)))


def test_corrupt_model_rejected(flat):
    right = flat.right.copy()
    first_internal = int(np.flatnonzero((flat.flags & 0x8) == 0)[0])
    right[first_internal] = 0  # child pointing backwards would loop forever
    with pytest.raises(ValueError):
        _treeeval.FlatEnsemble(flat.values, right, flat.feature, flat.flags, flat.roots, flat.n_features, 1.0)
    with pytest.raises(ValueError):
        _treeeval.FlatEnsemble(flat.values, flat.right, flat.feature, flat.flags, flat.roots, 2, 1.0)


def test_constant_model_single_leaf_trees():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 3))
    y = np.zeros(200)
    y[:100] = 1  # no signal at all with huge min_child_samples -> stump trees
    b = lgb.train({"objective": "binary", "min_data_in_leaf": 190, "verbose": -1}, lgb.Dataset(X, y), 3)
    f = flatten_booster(b)
    assert np.abs(CppEngine(f).predict(X) - b.predict(X)).max() < TOL


def test_categorical_splits_rejected():
    rng = np.random.default_rng(2)
    X = np.column_stack([rng.integers(0, 6, 500), rng.normal(size=500)])
    y = (X[:, 0] % 2 == 0).astype(int)
    b = lgb.train({"objective": "binary", "verbose": -1, "min_data_in_leaf": 5, "cat_smooth": 1},
                  lgb.Dataset(X, y, categorical_feature=[0]), 3)
    with pytest.raises(ValueError, match="categorical"):
        flatten_booster(b)


def test_non_binary_objective_rejected():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(100, 2))
    b = lgb.train({"objective": "regression", "verbose": -1}, lgb.Dataset(X, X[:, 0]), 2)
    with pytest.raises(ValueError, match="binary"):
        flatten_booster(b)


def test_probabilities_in_unit_interval(cpp, X):
    p = cpp.predict(X)
    assert ((p > 0) & (p < 1)).all() and not any(math.isnan(v) for v in p)
