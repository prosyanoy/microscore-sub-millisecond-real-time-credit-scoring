#!/usr/bin/env bash
# ~45 s terminal demo: start the API, load it with Locust for 30 s, show metrics + shadow summary.
# Usage: benchmarks/demo.sh   (from the repo root, with the venv at .venv)
set -euo pipefail
export MLFLOW_DISABLE_AGENT_HINT=1
BIN=${BIN:-.venv/bin}
PORT=${PORT:-8000}
DURATION=${DURATION:-30s}

echo "== starting API (first start trains + registers champion/challenger) =="
"$BIN/uvicorn" microscore.api:app --port "$PORT" --log-level warning &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 90); do
  kill -0 $SERVER 2>/dev/null || { echo "API failed to start (is port $PORT in use? try PORT=8010)"; exit 1; }
  curl -sf "localhost:$PORT/health" >/dev/null && break
  sleep 1
done
curl -s "localhost:$PORT/model"; echo

echo "== load test: 16 users, $DURATION =="
"$BIN/locust" -f benchmarks/locustfile.py --headless -u 16 -r 16 -t "$DURATION" \
  --host "http://127.0.0.1:$PORT" --only-summary 2>&1 | grep -E "^(Type|POST|\s+Aggregated|---)|Name" || true

echo "== engine-only latency recorded by the server (Prometheus histogram) =="
curl -s "localhost:$PORT/metrics" | grep -E "^microscore_(scoring_seconds_(sum|count)|rows_scored_total|shadow_rows_total)"
echo "== champion vs challenger (shadow mode) =="
curl -s "localhost:$PORT/shadow/summary"; echo
