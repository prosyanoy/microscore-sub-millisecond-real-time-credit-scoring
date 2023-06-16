"""Runtime configuration, read from MICROSCORE_* environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    mlflow_uri: str = "sqlite:///mlflow/mlflow.db"
    model_name: str = "microscore-credit"
    champion_alias: str = "champion"
    challenger_alias: str = "challenger"
    engine: str = "cpp"
    n_threads: int = 1
    shadow_enabled: bool = True
    shadow_db: str = "data/shadow.db"
    shadow_log_features: bool = True
    auto_seed: bool = True
    approve_below: float = 0.10
    decline_above: float = 0.25
    cache_dir: str = ".cache/models"
    max_batch: int = 10_000

    @classmethod
    def from_env(cls) -> "Settings":
        e = os.environ.get
        d = cls()
        return cls(
            mlflow_uri=e("MICROSCORE_MLFLOW_URI", d.mlflow_uri),
            model_name=e("MICROSCORE_MODEL_NAME", d.model_name),
            champion_alias=e("MICROSCORE_CHAMPION_ALIAS", d.champion_alias),
            challenger_alias=e("MICROSCORE_CHALLENGER_ALIAS", d.challenger_alias),
            engine=e("MICROSCORE_ENGINE", d.engine),
            n_threads=int(e("MICROSCORE_N_THREADS", d.n_threads)),
            shadow_enabled=_bool("MICROSCORE_SHADOW_ENABLED", d.shadow_enabled),
            shadow_db=e("MICROSCORE_SHADOW_DB", d.shadow_db),
            shadow_log_features=_bool("MICROSCORE_SHADOW_LOG_FEATURES", d.shadow_log_features),
            auto_seed=_bool("MICROSCORE_AUTO_SEED", d.auto_seed),
            approve_below=float(e("MICROSCORE_APPROVE_BELOW", d.approve_below)),
            decline_above=float(e("MICROSCORE_DECLINE_ABOVE", d.decline_above)),
            cache_dir=e("MICROSCORE_CACHE_DIR", d.cache_dir),
        )
