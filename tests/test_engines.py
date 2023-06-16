import numpy as np

from microscore.engines import LightGBMEngine, OnnxEngine


def test_onnx_matches_lightgbm(booster, X, tmp_path):
    from microscore.train import export_onnx

    export_onnx(booster, tmp_path / "m.onnx")
    e = OnnxEngine(tmp_path / "m.onnx")
    # ONNX evaluates features in float32, so parity is looser than for the C++ path.
    assert np.abs(e.predict(X) - booster.predict(X)).max() < 1e-4
    assert abs(e.predict_row(X[0]) - booster.predict(X[:1])[0]) < 1e-4


def test_lightgbm_engine_row_equals_batch(booster, X):
    e = LightGBMEngine(booster, num_threads=1)
    assert abs(e.predict_row(X[3]) - e.predict(X[:5])[3]) < 1e-12
