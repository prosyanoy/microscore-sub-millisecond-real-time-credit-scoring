PY ?= .venv/bin/python

.PHONY: setup train serve test bench shadow-report
setup:
	python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
train:
	MLFLOW_DISABLE_AGENT_HINT=1 $(PY) -m microscore.train
serve:
	MLFLOW_DISABLE_AGENT_HINT=1 .venv/bin/uvicorn microscore.api:app --port 8000
test:
	$(PY) -m pytest -q
bench:
	$(PY) benchmarks/run_benchmark.py
shadow-report:
	$(PY) -m microscore.shadow
