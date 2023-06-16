"""FastAPI scoring service: champion scoring, shadow challenger, Prometheus metrics."""
from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field

from . import registry
from .config import Settings
from .engines import CppEngine, Engine, LightGBMEngine, OnnxEngine
from .schema import Application, applications_to_matrix, row_to_dict
from .shadow import ShadowLogger


@dataclass
class LoadedModel:
    version: str
    alias: str
    engine: Engine
    metadata: dict


def build_engine(bundle: registry.ModelBundle, name: str, n_threads: int) -> Engine:
    if name == "cpp":
        return CppEngine(bundle.flat, n_threads)
    if name == "lightgbm":
        return LightGBMEngine(bundle.booster)
    if name == "onnx":
        return OnnxEngine(bundle.onnx_path)
    raise ValueError(f"unknown engine {name!r} (expected cpp, lightgbm or onnx)")


def load_model(settings: Settings, alias: str) -> Optional[LoadedModel]:
    version = registry.resolve_alias(settings.mlflow_uri, settings.model_name, alias)
    if version is None:
        return None
    bundle = registry.load_bundle(settings.mlflow_uri, settings.model_name, version, settings.cache_dir)
    return LoadedModel(version, alias, build_engine(bundle, settings.engine, settings.n_threads), bundle.metadata)


class ScoreResponse(BaseModel):
    request_id: str
    model_version: str
    default_probability: float = Field(description="Predicted probability of default")
    decision: str = Field(description="approve | review | decline")
    engine: str
    latency_ms: float


class BatchRequest(BaseModel):
    applications: list[Application] = Field(min_length=1)


class BatchItem(BaseModel):
    default_probability: float
    decision: str


class BatchResponse(BaseModel):
    request_id: str
    model_version: str
    engine: str
    latency_ms: float
    results: list[BatchItem]


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter("microscore_http_requests_total", "HTTP requests", ["path", "status"], registry=r)
        self.http_latency = Histogram(
            "microscore_http_request_seconds", "End-to-end HTTP latency", ["path"], registry=r,
            buckets=(.0005, .001, .0025, .005, .01, .025, .05, .1, .25, 1),
        )
        self.score_latency = Histogram(
            "microscore_scoring_seconds", "Model evaluation latency", ["engine"], registry=r,
            buckets=(.00001, .00005, .0001, .00025, .0005, .001, .0025, .005, .01, .05),
        )
        self.rows = Counter("microscore_rows_scored_total", "Applications scored", registry=r)
        self.decisions = Counter("microscore_decisions_total", "Decisions issued", ["decision"], registry=r)
        self.shadow_rows = Counter("microscore_shadow_rows_total", "Rows scored by the challenger", registry=r)
        self.shadow_errors = Counter("microscore_shadow_errors_total", "Challenger failures", registry=r)
        self.model_info = Gauge("microscore_model_info", "Loaded models", ["role", "version"], registry=r)


class MetricsMiddleware:
    """Pure-ASGI middleware (cheaper than BaseHTTPMiddleware) recording per-route metrics."""

    def __init__(self, app, metrics: Metrics):
        self.app, self.m = app, metrics

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        start, status = time.perf_counter(), 500

        async def wrapped(msg):
            nonlocal status
            if msg["type"] == "http.response.start":
                status = msg["status"]
            await send(msg)

        try:
            await self.app(scope, receive, wrapped)
        finally:
            route = scope.get("route")
            path = getattr(route, "path", None) or "unmatched"
            self.m.requests.labels(path, str(status)).inc()
            self.m.http_latency.labels(path).observe(time.perf_counter() - start)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    metrics = Metrics()
    state: dict = {"champion": None, "challenger": None, "shadow": None}

    def decide(p: float) -> str:
        if p < settings.approve_below:
            return "approve"
        return "decline" if p > settings.decline_above else "review"

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if settings.auto_seed:
            from .train import seed_registry

            seed_registry(settings)
        state["champion"] = load_model(settings, settings.champion_alias)
        if state["champion"] is None:
            raise RuntimeError(
                f"no model with alias '{settings.champion_alias}' in {settings.mlflow_uri}; "
                "run `python -m microscore.train` or set MICROSCORE_AUTO_SEED=1"
            )
        metrics.model_info.labels("champion", state["champion"].version).set(1)
        if settings.shadow_enabled:
            ch = load_model(settings, settings.challenger_alias)
            if ch is not None:
                state["challenger"] = ch
                state["shadow"] = ShadowLogger(settings.shadow_db)
                metrics.model_info.labels("challenger", ch.version).set(1)
        yield
        if state["shadow"] is not None:
            state["shadow"].flush()
            state["shadow"].close()

    app = FastAPI(title="MicroScore", version="0.1.0", lifespan=lifespan)
    app.add_middleware(MetricsMiddleware, metrics=metrics)
    app.state.microscore = state
    app.state.metrics = metrics

    def champion() -> LoadedModel:
        m = state["champion"]
        if m is None:
            raise HTTPException(503, "model not loaded")
        return m

    def shadow_task(request_id: str, X: np.ndarray, champion_scores: np.ndarray, champ_version: str):
        """Runs after the response is sent; failures never affect callers."""
        ch: Optional[LoadedModel] = state["challenger"]
        if ch is None:
            return
        try:
            scores = ch.engine.predict(X)
            feats = [row_to_dict(r) for r in X] if settings.shadow_log_features else None
            state["shadow"].log(request_id, champ_version, ch.version, champion_scores, scores, feats)
            metrics.shadow_rows.inc(len(X))
        except Exception:  # pragma: no cover - defensive
            metrics.shadow_errors.inc()

    @app.post("/score", response_model=ScoreResponse)
    async def score(application: Application, request: Request, background: BackgroundTasks):
        m = champion()
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        x = np.asarray(application.to_row(), dtype=np.float64)
        t0 = time.perf_counter()
        p = m.engine.predict_row(x)
        dt = time.perf_counter() - t0
        metrics.score_latency.labels(m.engine.name).observe(dt)
        d = decide(p)
        metrics.rows.inc()
        metrics.decisions.labels(d).inc()
        if state["challenger"] is not None:
            background.add_task(shadow_task, request_id, x.reshape(1, -1), np.array([p]), m.version)
        return ScoreResponse(request_id=request_id, model_version=m.version, default_probability=p,
                             decision=d, engine=m.engine.name, latency_ms=dt * 1000)

    @app.post("/score/batch", response_model=BatchResponse)
    async def score_batch(req: BatchRequest, request: Request, background: BackgroundTasks):
        if len(req.applications) > settings.max_batch:
            raise HTTPException(413, f"batch too large (max {settings.max_batch})")
        m = champion()
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        X = applications_to_matrix(req.applications)
        t0 = time.perf_counter()
        p = m.engine.predict(X)
        dt = time.perf_counter() - t0
        metrics.score_latency.labels(m.engine.name).observe(dt)
        decisions = [decide(float(v)) for v in p]
        metrics.rows.inc(len(p))
        for d in decisions:
            metrics.decisions.labels(d).inc()
        if state["challenger"] is not None:
            background.add_task(shadow_task, request_id, X, p, m.version)
        return BatchResponse(
            request_id=request_id, model_version=m.version, engine=m.engine.name, latency_ms=dt * 1000,
            results=[BatchItem(default_probability=float(v), decision=d) for v, d in zip(p, decisions)],
        )

    @app.get("/health")
    async def health():
        return {"status": "ok", "champion_version": champion().version}

    @app.get("/model")
    async def model_info():
        m = champion()
        ch = state["challenger"]
        return {
            "model_name": settings.model_name,
            "champion": {"version": m.version, "alias": m.alias, "engine": m.engine.name,
                         "metrics": m.metadata.get("metrics")},
            "challenger": None if ch is None else {"version": ch.version, "alias": ch.alias,
                                                   "metrics": ch.metadata.get("metrics")},
        }

    @app.get("/shadow/summary")
    def shadow_summary():
        if state["shadow"] is None:
            raise HTTPException(404, "shadow mode is not active")
        return state["shadow"].summary(settings.approve_below, settings.decline_above)

    @app.get("/metrics")
    async def prometheus():
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
