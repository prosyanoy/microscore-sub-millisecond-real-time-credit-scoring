import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import pytest

from microscore.config import Settings
from microscore.train import build_bundle, train_booster, PARAMS

SMALL = dict(PARAMS["champion"], n_estimators=40)


@pytest.fixture(scope="session")
def trained():
    booster, metrics, (X, y) = train_booster(SMALL, n_rows=8000)
    return booster, metrics, X


@pytest.fixture(scope="session")
def booster(trained):
    return trained[0]


@pytest.fixture(scope="session")
def X(trained):
    return trained[2].to_numpy()


@pytest.fixture(scope="session")
def settings(tmp_path_factory):
    """A private MLflow store seeded with a champion and a (different) challenger."""
    from microscore.registry import register_bundle

    root = tmp_path_factory.mktemp("store")
    s = Settings(
        mlflow_uri=f"sqlite:///{root}/mlflow.db",
        shadow_db=str(root / "shadow.db"),
        cache_dir=str(root / "cache"),
        auto_seed=False,
    )
    for role, params in (("champion", SMALL), ("challenger", dict(SMALL, num_leaves=15, random_state=9))):
        b, m, _ = train_booster(params, n_rows=8000)
        bundle = build_bundle(b, root / f"bundle-{role}", {"metrics": m})
        alias = s.champion_alias if role == "champion" else s.challenger_alias
        register_bundle(s.mlflow_uri, s.model_name, bundle, params, m, alias)
    return s
