"""Reproducible engine benchmark: LightGBM (Python) vs ONNX Runtime vs custom C++ evaluator.

Trains the champion model on seeded synthetic data (no registry needed), then measures
single-row latency (p50/p90/p99/p99.9) and batch throughput for every engine.

    python benchmarks/run_benchmark.py            # full run
    python benchmarks/run_benchmark.py --quick    # smoke run
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import lightgbm as lgb
import numpy as np

from microscore.data import make_dataset
from microscore.engines import CppEngine, LightGBMEngine, OnnxEngine
from microscore.flatten import flatten_booster
from microscore.train import PARAMS, export_onnx, train_booster

RESULTS = Path(__file__).parent / "results"


def single_row_latency(engine, rows: np.ndarray, n_calls: int, warmup: int) -> dict:
    n = len(rows)
    for i in range(warmup):
        engine.predict_row(rows[i % n])
    samples = np.empty(n_calls, dtype=np.int64)
    clock = time.perf_counter_ns
    for i in range(n_calls):
        x = rows[i % n]
        t = clock()
        engine.predict_row(x)
        samples[i] = clock() - t
    us = samples / 1000.0
    q = lambda p: float(np.percentile(us, p))
    return {"p50_us": q(50), "p90_us": q(90), "p99_us": q(99), "p999_us": q(99.9),
            "mean_us": float(us.mean()), "calls": n_calls}


def batch_throughput(engine, X: np.ndarray, batch: int, seconds: float) -> dict:
    Xb = np.ascontiguousarray(X[:batch])
    for _ in range(3):
        engine.predict(Xb)
    lat, rows, end = [], 0, time.perf_counter() + seconds
    while time.perf_counter() < end:
        t = time.perf_counter()
        engine.predict(Xb)
        lat.append(time.perf_counter() - t)
        rows += batch
    total = sum(lat)
    return {"batch": batch, "rows_per_s": rows / total, "p50_ms": statistics.median(lat) * 1e3,
            "p99_ms": float(np.percentile(lat, 99)) * 1e3}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--calls", type=int, default=20_000)
    ap.add_argument("--seconds", type=float, default=2.0)
    args = ap.parse_args()
    if args.quick:
        args.calls, args.seconds = 2_000, 0.3

    booster, metrics, (Xte, _) = train_booster(PARAMS["champion"])
    X = Xte.to_numpy().astype(np.float64)
    flat = flatten_booster(booster)

    with tempfile.TemporaryDirectory() as tmp:
        export_onnx(booster, Path(tmp) / "m.onnx")
        engines = {
            "lightgbm": LightGBMEngine(booster),
            "lightgbm-1t": LightGBMEngine(booster, num_threads=1),
            "onnxruntime": OnnxEngine(Path(tmp) / "m.onnx", num_threads=1),
            "cpp": CppEngine(flat, 1),
        }
        if args.threads > 1:
            engines[f"cpp-{args.threads}t"] = CppEngine(flat, args.threads)

        ref = booster.predict(X)
        out = {
            "environment": {"python": platform.python_version(), "cpu": platform.processor() or platform.machine(),
                            "cores": os.cpu_count(), "lightgbm": lgb.__version__, "numpy": np.__version__,
                            "trees": flat.n_trees, "nodes": flat.n_nodes, "auc": metrics["auc"]},
            "max_abs_diff_vs_lightgbm": {n: float(np.abs(e.predict(X) - ref).max()) for n, e in engines.items()},
            "single_row": {}, "batch": {},
        }
        for name, e in engines.items():
            if "t" in name.split("-")[-1] and name.startswith("cpp-"):
                continue  # multithreading is irrelevant for one row
            print(f"single-row {name} ...", flush=True)
            out["single_row"][name] = single_row_latency(e, X, args.calls, 500)
        for batch in (1, 64, 1024):
            out["batch"][str(batch)] = {}
            for name, e in engines.items():
                print(f"batch={batch} {name} ...", flush=True)
                out["batch"][str(batch)][name] = batch_throughput(e, X, batch, args.seconds)

    md = render_markdown(out)
    if args.quick:  # smoke runs must not overwrite the reference results
        print(md)
        return
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "benchmark.json").write_text(json.dumps(out, indent=2))
    (RESULTS / "RESULTS.md").write_text(md)
    print(md)


def render_markdown(out: dict) -> str:
    env = out["environment"]
    lines = [f"# Benchmark results\n",
             f"{env['trees']} trees / {env['nodes']} nodes, {env['cores']} cores, "
             f"Python {env['python']}, LightGBM {env['lightgbm']}. Numbers are machine-specific.\n",
             "## Single-row latency (microseconds, in-process, 1 row per call)\n",
             "| engine | p50 | p90 | p99 | p99.9 | mean | max abs diff vs LightGBM |", "|---|---|---|---|---|---|---|"]
    base = out["single_row"]["lightgbm"]["p50_us"]
    for n, r in out["single_row"].items():
        lines.append(f"| {n} | {r['p50_us']:.1f} | {r['p90_us']:.1f} | {r['p99_us']:.1f} | {r['p999_us']:.1f} | "
                     f"{r['mean_us']:.1f} | {out['max_abs_diff_vs_lightgbm'][n]:.1e} |")
    cpp = out["single_row"]["cpp"]["p50_us"]
    lines.append(f"\nC++ p50 is {base / cpp:.0f}x lower than LightGBM's default Python path.\n")
    for batch, rows in out["batch"].items():
        lines += [f"## Batch size {batch}\n", "| engine | rows/s | p50 batch ms | p99 batch ms |", "|---|---|---|---|"]
        for n, r in rows.items():
            lines.append(f"| {n} | {r['rows_per_s']:,.0f} | {r['p50_ms']:.3f} | {r['p99_ms']:.3f} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
