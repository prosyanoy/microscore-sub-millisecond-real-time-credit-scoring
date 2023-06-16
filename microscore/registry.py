"""Model bundles and the local MLflow registry (SQLite backend, aliases for roles)."""
from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import mlflow
from mlflow import MlflowClient

from .flatten import FlatModel

EXPERIMENT = "microscore"
BUNDLE_DIR = "model_bundle"


@dataclass
class ModelBundle:
    """Everything needed to serve one registered model version."""

    version: str
    path: Path
    metadata: dict

    @property
    def booster(self) -> lgb.Booster:
        return lgb.Booster(model_file=str(self.path / "model.txt"))

    @property
    def flat(self) -> FlatModel:
        return FlatModel.load(self.path / "flat.npz")

    @property
    def onnx_path(self) -> Path:
        return self.path / "model.onnx"


def _client(tracking_uri: str) -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri)
    if tracking_uri.startswith("sqlite:///"):
        db = Path(tracking_uri[len("sqlite:///"):])
        db.parent.mkdir(parents=True, exist_ok=True)
    return MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)


def _experiment_id(client: MlflowClient, tracking_uri: str) -> str:
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is not None:
        return exp.experiment_id
    if tracking_uri.startswith("sqlite:///"):
        artifact_root = (Path(tracking_uri[len("sqlite:///"):]).resolve().parent / "artifacts").as_uri()
    else:
        artifact_root = None
    return client.create_experiment(EXPERIMENT, artifact_location=artifact_root)


def register_bundle(
    tracking_uri: str,
    model_name: str,
    bundle_dir: Path,
    params: dict,
    metrics: dict,
    alias: str | None,
) -> str:
    """Log `bundle_dir` as run artifacts, register a model version, optionally set an alias."""
    client = _client(tracking_uri)
    exp_id = _experiment_id(client, tracking_uri)
    with mlflow.start_run(experiment_id=exp_id, run_name=f"{model_name}-{alias or 'train'}") as run:
        mlflow.log_params({k: v for k, v in params.items()})
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(bundle_dir), BUNDLE_DIR)
        run_id = run.info.run_id
    try:
        client.create_registered_model(model_name)
    except mlflow.exceptions.MlflowException:
        pass  # already exists
    mv = client.create_model_version(
        model_name, source=f"{client.get_run(run_id).info.artifact_uri}/{BUNDLE_DIR}", run_id=run_id
    )
    if alias:
        client.set_registered_model_alias(model_name, alias, mv.version)
    return str(mv.version)


def resolve_alias(tracking_uri: str, model_name: str, alias: str) -> str | None:
    """Return the version string the alias points at, or None."""
    client = _client(tracking_uri)
    try:
        return str(client.get_model_version_by_alias(model_name, alias).version)
    except mlflow.exceptions.MlflowException:
        return None


def load_bundle(tracking_uri: str, model_name: str, version: str, cache_dir: str) -> ModelBundle:
    """Download (and cache by version) the bundle of a registered model version."""
    client = _client(tracking_uri)
    mv = client.get_model_version(model_name, version)
    dest = Path(cache_dir) / model_name / str(version)
    if not (dest / "metadata.json").exists():
        tmp = Path(tempfile.mkdtemp(prefix="bundle-"))
        try:
            local = mlflow.artifacts.download_artifacts(
                run_id=mv.run_id, artifact_path=BUNDLE_DIR, dst_path=str(tmp)
            )
            if dest.exists():
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(local, dest)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    metadata = json.loads((dest / "metadata.json").read_text())
    return ModelBundle(version=str(version), path=dest, metadata=metadata)
