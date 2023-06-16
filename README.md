# MicroScore: Sub-Millisecond Real-Time Credit Scoring

A real-time loan-scoring service that serves a gradient-boosted model through a hand-optimized C++ tree-ensemble evaluator and is benchmarked against the standard Python path.

> **~5x lower p99 than single-thread LightGBM (3x lower than ONNX Runtime), bit-compatible predictions (max abs diff 0.0).**
> 14 µs p50 / 31 µs p99 per scored row vs 97 / 162 µs for LightGBM and 49 / 100 µs for ONNX Runtime.

| engine (300 trees, 1 row/call) | p50 µs | p99 µs | p99 vs C++ | max abs diff vs LightGBM |
|---|---|---|---|---|
| LightGBM, 1 thread | 96.7 | 161.9 | 5.2x | 0 |
| LightGBM, default threads | 91.8 | 3480.1 | 113x (OpenMP jitter, noisy) | 0 |
| ONNX Runtime | 48.9 | 99.8 | 3.2x | 3.2e-07 (float32 inputs) |
| **C++ evaluator (this repo)** | **14.3** | **30.9** | 1x | **0** |

In-process, 2-core shared VM, single run of `python benchmarks/run_benchmark.py`; absolute numbers are machine-specific, so re-run on yours (full tables incl. batch throughput: `benchmarks/results/RESULTS.md`).

**CV line:** Built a real-time credit scoring service with a custom C++ GBDT evaluator (pybind11), cutting p99 latency by ~5x while matching LightGBM output.

## Features

- **LightGBM credit model** trained on seeded synthetic loan data (13 features, non-linear defaults, missing values).
- **Flat tree format**: trees are exported into six contiguous arrays (`.npz`), depth-first so the left child is always the next node. The C++ side packs them into 16-byte nodes (4 per cache line).
- **C++17 evaluator** (`cpp/treeeval.cpp`, pybind11): single-row and batch modes, optional multithreading (GIL released), 8-row lock-step traversal in batch mode, LightGBM-exact missing/zero handling, validation of corrupt model files. Output matches LightGBM to **0.0 absolute difference** on the test data (requirement: 1e-6; asserted in tests).
- **FastAPI service**: pydantic-validated `/score` and `/score/batch`, model versions loaded from a local **MLflow registry** via `champion`/`challenger` aliases, Prometheus `/metrics`, `/health`, `/model`.
- **Benchmark suite** comparing LightGBM (Python, default and 1 thread), **ONNX Runtime** and the C++ evaluator: p50/p90/p99/p99.9 single-row latency and batch throughput, plus a Locust file for HTTP load.
- **Shadow mode**: the challenger scores every request *after* the response is sent; both scores (and optionally the features) are written to SQLite and summarised at `/shadow/summary` or with `python -m microscore.shadow`.

## Architecture

```
train.py ──► LightGBM booster ──► bundle { model.txt, flat.npz, model.onnx, metadata.json }
                                         │ logged as MLflow run artifacts + registered model version
                                         ▼
                       MLflow registry (SQLite)  alias champion ─► v1   alias challenger ─► v2
                                         │
FastAPI (api.py) ── loads both at startup ──► Engine (cpp | lightgbm | onnx)
   POST /score ─► champion engine ─► response
                       └─ background task: challenger engine ─► ShadowLogger queue ─► SQLite writer thread
   GET /metrics (Prometheus)   GET /shadow/summary   GET /model   GET /health
```

```
cpp/treeeval.cpp        C++17 evaluator + pybind11 bindings (module microscore._treeeval)
microscore/             schema, data, flatten, engines, registry, train, shadow, api, config
benchmarks/             run_benchmark.py, locustfile.py, results/RESULTS.md
tests/                  pytest suite (parity, missing values, API, registry, shadow)
Dockerfile, docker-compose.yml, Makefile, .env.example
```

## Setup

Requires Python ≥ 3.10 and a C++17 compiler (g++/clang).

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"     # builds the C++ extension
```

## Run

```bash
export MLFLOW_DISABLE_AGENT_HINT=1                 # optional: silences an MLflow notice
.venv/bin/python -m microscore.train               # optional: first server start seeds models itself
.venv/bin/uvicorn microscore.api:app --port 8000

curl -s localhost:8000/score -H 'content-type: application/json' -d '{
  "age": 41, "annual_income": 72000, "loan_amount": 15000, "term_months": 36,
  "interest_rate": 11.5, "debt_to_income": 0.31, "credit_utilization": 0.42,
  "open_accounts": 7, "delinquencies_2y": 0, "credit_history_months": 180,
  "employment_years": 6.5, "home_ownership": 1, "purpose": 0}'
curl -s localhost:8000/metrics | grep microscore_
curl -s localhost:8000/shadow/summary
```

With `MICROSCORE_AUTO_SEED=1` (the default) the first start trains and registers a champion and a challenger in `./mlflow/` (SQLite store + local artifacts), so no setup or secrets are needed. To promote a new model, register a version and move the alias; restart to pick it up. Config is entirely via `MICROSCORE_*` variables; see `.env.example`. Switch engines with `MICROSCORE_ENGINE=cpp|lightgbm|onnx`.

Docker: `docker compose up --build` (API on :8000, state in a volume); `docker compose --profile load up loadtest` runs Locust against it. *(Note: the Dockerfile could not be built in the authoring sandbox because the Docker daemon was not accessible, so it is untested.)*

HTTP load test (server running): `.venv/bin/locust -f benchmarks/locustfile.py --headless -u 32 -r 32 -t 20s --host http://127.0.0.1:8000`.

## Test

```bash
.venv/bin/python -m pytest -q       # 37 tests, ~5 s
```

## Benchmarks

```bash
.venv/bin/python benchmarks/run_benchmark.py     # writes benchmarks/results/RESULTS.md (+ benchmark.json); --quick = smoke run, writes nothing
```

Batch of 1024 rows, single thread: C++ ≈ 99k rows/s vs ONNX Runtime ≈ 96k vs LightGBM ≈ 38k (C++ and ONNX are roughly on par; the C++ advantage is in single-row and small-batch latency).

**Why it is faster** (what the profile points to; I did not run a hardware-counter profiler, so the cache/branch claims are design rationale, not measurements):
- *Python/dispatch overhead.* A one-row `Booster.predict` spends most of its ~90 µs in input validation, conversion and OpenMP setup; the tree walk itself is a few µs. `predict_one` has none of this.
- *Memory layout.* 16-byte nodes, all trees contiguous, left child = next node: ~18k nodes ≈ 290 KB for the demo model, so most of it stays in L2 and the hot top levels in L1.
- *Batching.* Batch mode walks 8 rows through each tree in lock-step so independent traversals overlap their load latency and branch resolution (≈ +15-20% rows/s over the plain loop here). Threads engage only above 256 rows/thread.
- *Not done:* SIMD/AVX gather traversal and branch-free quantised (QuickScorer-style) evaluation; they would help large batches most.

## Demo (load test, ~45 s)

```bash
PORT=8010 benchmarks/demo.sh     # starts the API, runs Locust for 30 s, prints metrics and the shadow summary
```

Captured output (2-core VM where Locust and the server compete for the same CPUs; first start also trains the demo models, so total wall time was ~55 s):

```
== load test: 16 users, 30s ==
Type     Name              # reqs   # fails |  Avg   Min   Max   Med |  req/s
POST     /score             14904    0(0.00%) |   28     4   132    27 | 504.41
POST     /score/batch        1484    0(0.00%) |   29     8   108    29 |  50.22
         Aggregated         16388    0(0.00%) |   28     4   132    27 | 554.63
Aggregated percentiles (ms): 50% 27  90% 39  95% 46  99% 66  99.9% 100
== engine-only latency recorded by the server (Prometheus histogram) ==
microscore_scoring_seconds_count{engine="cpp"} 16388
microscore_scoring_seconds_sum{engine="cpp"} 1.88        # mean 115 µs per request, incl. 50-row batches
microscore_rows_scored_total 89104
== champion vs challenger (shadow mode) ==
{"count":89104,"dropped":0,"mean_abs_diff":0.0356,"decision_disagreement_rate": ...}
```

HTTP latencies (tens of ms) are dominated by Locust, JSON and Python request handling on a shared 2-core box, not by the model; the evaluator's share is the ~0.1 ms in the histogram. To make a recording, run the script under `asciinema rec` (not included in the repo).

## Correctness testing and trade-offs

**How correctness is established** (`tests/test_flat_evaluator.py`, `tests/test_registry_and_api.py`):
- C++ batch, single-row, raw-score and multithreaded outputs vs `lgb.Booster.predict` on held-out data, asserted to `< 1e-6` (observed: exactly 0.0); multithreaded output must be bit-identical to single-thread.
- Random NaN (20%) and exact-zero (10%) injections, because LightGBM's missing-value routing is the easiest thing to get subtly wrong.
- A slow pure-Python evaluator of the same flat format as an independent oracle, plus `.npz` save/load round trips.
- Edge cases: stump (single-leaf) trees, rejected categorical splits and non-binary objectives, shape errors, and corrupt child indices (no infinite loops or out-of-bounds reads).
- The API test compares the served score to LightGBM's score for the registered model version; ONNX parity is checked at `1e-4` since it uses float32.

**When this is not worth it:**
- If your p99 budget is tens of milliseconds, network and serialisation dominate (see the demo above), and `Booster.predict` or ONNX Runtime is fine. The win here is ~80 µs per row.
- Large offline batches: ONNX Runtime and LightGBM are already competitive and better maintained.
- Models that need categorical splits, multiclass/ranking objectives, or frequent architecture changes: the evaluator only supports numeric splits and binary objectives, and every new LightGBM feature is a maintenance burden you own.
- Regulated settings often value a well-known runtime over bespoke code; the parity tests mitigate but do not remove that cost.
- It only pays off when per-row latency/tail latency or CPU cost per request genuinely matters (e.g. scoring thousands of candidate offers inside one user request).

## Key design decisions

- **Flat DFS layout, left child = next node.** Halves the pointer data per node and keeps hot paths sequential in memory; leaf values live inline in the node array.
- **Exactness over approximation.** Thresholds and leaf values stay `float64` and are summed in tree order, so results are bit-identical to LightGBM; ONNX Runtime necessarily uses float32 inputs and drifts ~1e-7.
- **Missing values follow LightGBM semantics** (None/Zero/NaN missing types and default direction), covered by tests with random NaNs and zeros.
- **Only numeric splits and binary objectives** are supported; the exporter raises a clear error for categorical splits (categorical-like features are encoded as ordinal integers).
- **Safe loading.** The C++ constructor checks that children lie strictly after their parent and features are in range, so a corrupted `.npz` cannot loop forever or read out of bounds.
- **MLflow aliases, not stages** (stages are deprecated). Bundles are logged as run artifacts, downloaded once per version into a local cache. The SQLite backend keeps everything embedded.
- **Shadow scoring is off the response path** (after the response via a background task, then a queued SQLite writer thread). The queue is bounded: under overload shadow rows are dropped and counted rather than slowing the champion. Challenger failures never reach callers.
- **Multithreading is batch-only and opt-in** (`n_threads`); it only engages with ≥256 rows per thread because thread start-up costs more than scoring a small batch.
- **No `-march=native` by default** so wheels/images are portable; set `MICROSCORE_NATIVE=1` when building for a known host.
- Decisions: `approve` below 0.10 PD, `decline` above 0.25, else `review` (configurable). The synthetic data are for demonstration only; the model metrics (AUC ≈ 0.82) say nothing about real credit risk.
