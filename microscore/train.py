"""Train LightGBM credit models, export flat/ONNX artifacts and register them in MLflow."""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from .config import Settings
from .data import make_dataset
from .flatten import flatten_booster
from .registry import register_bundle, resolve_alias
from .schema import FEATURES

PARAMS = {
    "champion": dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=40,
                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=1),
    "challenger": dict(n_estimators=500, learning_rate=0.03, num_leaves=63, min_child_samples=30,
                       subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0,
                       random_state=2),
}


def train_booster(params: dict, n_rows: int = 60_000, data_seed: int = 0):
    """Train on synthetic data; returns (booster, metrics, holdout X/y)."""
    X, y = make_dataset(n_rows, data_seed)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    model = lgb.LGBMClassifier(objective="binary", verbose=-1, n_jobs=1, **params)
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    order = np.argsort(p)
    cum_pos = np.cumsum(yte[order]) / yte.sum()
    cum_neg = np.cumsum(1 - yte[order]) / (1 - yte).sum()
    metrics = {
        "auc": float(roc_auc_score(yte, p)),
        "logloss": float(log_loss(yte, p)),
        "ks": float(np.max(np.abs(cum_pos - cum_neg))),
        "default_rate": float(y.mean()),
    }
    return model.booster_, metrics, (Xte, yte)


def export_onnx(booster: lgb.Booster, path: Path) -> None:
    """Convert to ONNX (float32 inputs, probabilities output) for the comparison engine."""
    from onnxmltools import convert_lightgbm
    from onnxmltools.convert.common.data_types import FloatTensorType

    onx = convert_lightgbm(
        booster,
        initial_types=[("input", FloatTensorType([None, len(FEATURES)]))],
        target_opset=15,
        zipmap=False,
    )
    path.write_bytes(onx.SerializeToString())


def build_bundle(booster: lgb.Booster, out: Path, metadata: dict, with_onnx: bool = True) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(out / "model.txt"))
    flatten_booster(booster).save(out / "flat.npz")
    if with_onnx:
        export_onnx(booster, out / "model.onnx")
    (out / "metadata.json").write_text(json.dumps({**metadata, "features": FEATURES}, indent=2))
    return out


def train_and_register(role: str, settings: Settings, n_rows: int = 60_000) -> str:
    params = PARAMS[role]
    booster, metrics, _ = train_booster(params, n_rows)
    with tempfile.TemporaryDirectory() as tmp:
        build_bundle(booster, Path(tmp), {"role_at_training": role, "metrics": metrics, "params": params})
        alias = settings.champion_alias if role == "champion" else settings.challenger_alias
        version = register_bundle(settings.mlflow_uri, settings.model_name, Path(tmp), params, metrics, alias)
    print(f"registered {settings.model_name} v{version} as '{alias}': "
          + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return version


def seed_registry(settings: Settings, n_rows: int = 60_000) -> None:
    """Idempotently make sure champion and challenger aliases exist."""
    if resolve_alias(settings.mlflow_uri, settings.model_name, settings.champion_alias) is None:
        train_and_register("champion", settings, n_rows)
    if resolve_alias(settings.mlflow_uri, settings.model_name, settings.challenger_alias) is None:
        train_and_register("challenger", settings, n_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train and register MicroScore models")
    ap.add_argument("--role", choices=["champion", "challenger", "both"], default="both")
    ap.add_argument("--rows", type=int, default=60_000)
    args = ap.parse_args()
    s = Settings.from_env()
    for role in (["champion", "challenger"] if args.role == "both" else [args.role]):
        train_and_register(role, s, args.rows)


if __name__ == "__main__":
    main()
