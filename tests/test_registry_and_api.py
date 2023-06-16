import math
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

from microscore import registry
from microscore.api import create_app
from microscore.data import make_dataset
from microscore.engines import CppEngine
from microscore.schema import FEATURES

GOOD = {
    "age": 41, "annual_income": 72000, "loan_amount": 15000, "term_months": 36, "interest_rate": 11.5,
    "debt_to_income": 0.31, "credit_utilization": 0.42, "open_accounts": 7, "delinquencies_2y": 0,
    "credit_history_months": 180, "employment_years": 6.5, "home_ownership": 1, "purpose": 0,
}


def test_registry_aliases_and_bundle(settings):
    champ = registry.resolve_alias(settings.mlflow_uri, settings.model_name, "champion")
    chall = registry.resolve_alias(settings.mlflow_uri, settings.model_name, "challenger")
    assert champ != chall
    assert registry.resolve_alias(settings.mlflow_uri, settings.model_name, "nope") is None
    b = registry.load_bundle(settings.mlflow_uri, settings.model_name, champ, settings.cache_dir)
    assert b.metadata["features"] == FEATURES
    for f in ("model.txt", "flat.npz", "model.onnx"):
        assert (b.path / f).exists()
    X, _ = make_dataset(300, seed=5)
    got = CppEngine(b.flat).predict(X.to_numpy())
    assert np.abs(got - b.booster.predict(X.to_numpy())).max() < 1e-6


@pytest.fixture()
def client(settings, tmp_path):
    s = type(settings)(**{**settings.__dict__, "shadow_db": str(tmp_path / "shadow.db")})
    with TestClient(create_app(s)) as c:
        yield c


def test_score_matches_model(client, settings):
    r = client.post("/score", json=GOOD, headers={"X-Request-ID": "abc"})
    assert r.status_code == 200
    body = r.json()
    assert body["request_id"] == "abc" and body["engine"] == "cpp"
    assert 0 < body["default_probability"] < 1
    assert body["decision"] in {"approve", "review", "decline"}
    v = registry.resolve_alias(settings.mlflow_uri, settings.model_name, "champion")
    assert body["model_version"] == v
    bundle = registry.load_bundle(settings.mlflow_uri, settings.model_name, v, settings.cache_dir)
    row = np.array([[GOOD[f] for f in FEATURES]], dtype=float)
    assert body["default_probability"] == pytest.approx(bundle.booster.predict(row)[0], abs=1e-6)


def test_missing_optional_field_is_scored(client):
    payload = {k: v for k, v in GOOD.items() if k != "employment_years"}
    assert client.post("/score", json=payload).status_code == 200


@pytest.mark.parametrize("patch", [{"age": 5}, {"credit_utilization": -1}, {"purpose": 9}, {"bogus": 1}, {"term_months": "x"}])
def test_validation_errors(client, patch):
    assert client.post("/score", json={**GOOD, **patch}).status_code == 422


def test_required_field_missing(client):
    payload = {k: v for k, v in GOOD.items() if k != "age"}
    assert client.post("/score", json=payload).status_code == 422


def test_batch_consistent_with_single(client):
    apps = [dict(GOOD, loan_amount=a) for a in (1000, 20000, 90000)]
    r = client.post("/score/batch", json={"applications": apps})
    assert r.status_code == 200
    got = [x["default_probability"] for x in r.json()["results"]]
    single = [client.post("/score", json=a).json()["default_probability"] for a in apps]
    assert got == pytest.approx(single, abs=1e-12)
    assert client.post("/score/batch", json={"applications": []}).status_code == 422


def test_metrics_endpoint(client):
    client.post("/score", json=GOOD)
    client.post("/score", json={**GOOD, "age": 1})
    text = client.get("/metrics").text
    assert 'microscore_http_requests_total{path="/score",status="200"}' in text
    assert 'microscore_http_requests_total{path="/score",status="422"}' in text
    assert "microscore_scoring_seconds_bucket" in text
    assert 'microscore_model_info{role="champion"' in text


def test_shadow_logging(client, tmp_path):
    for la in (5000, 30000):
        client.post("/score", json=dict(GOOD, loan_amount=la), headers={"X-Request-ID": f"r{la}"})
    client.post("/score/batch", json={"applications": [GOOD, GOOD]})
    summary = client.get("/shadow/summary").json()
    assert summary["count"] == 4
    assert summary["mean_abs_diff"] >= 0
    info = client.get("/model").json()
    assert info["challenger"]["version"] != info["champion"]["version"]
    rows = sqlite3.connect(tmp_path / "shadow.db").execute(
        "SELECT champion_version, challenger_version, champion_score, challenger_score, features "
        "FROM shadow_log WHERE request_id='r5000'").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == info["champion"]["version"] and rows[0][1] == info["challenger"]["version"]
    assert not math.isnan(rows[0][3]) and '"loan_amount": 5000' in rows[0][4]


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_startup_fails_without_champion(tmp_path, settings):
    s = type(settings)(**{**settings.__dict__, "mlflow_uri": f"sqlite:///{tmp_path}/empty.db",
                          "shadow_db": str(tmp_path / "x.db")})
    with pytest.raises(RuntimeError, match="no model"):
        with TestClient(create_app(s)):
            pass


@pytest.mark.parametrize("engine", ["lightgbm", "onnx"])
def test_alternate_engines(settings, tmp_path, engine):
    s = type(settings)(**{**settings.__dict__, "engine": engine, "shadow_enabled": False})
    with TestClient(create_app(s)) as c:
        body = c.post("/score", json=GOOD).json()
        assert body["engine"].startswith(engine[:4]) or body["engine"] == "onnxruntime"
        assert 0 < body["default_probability"] < 1
