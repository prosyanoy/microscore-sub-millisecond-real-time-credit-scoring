FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY pyproject.toml setup.py ./
COPY cpp ./cpp
COPY microscore ./microscore
RUN pip install --no-cache-dir build && python -m build --wheel --outdir /wheels .

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && useradd -m app
COPY --from=build /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl && rm -rf /wheels
USER app
WORKDIR /home/app
ENV MLFLOW_DISABLE_AGENT_HINT=1 MICROSCORE_MLFLOW_URI=sqlite:////home/app/state/mlflow/mlflow.db \
    MICROSCORE_SHADOW_DB=/home/app/state/shadow.db MICROSCORE_CACHE_DIR=/home/app/state/cache
EXPOSE 8000
# First start trains and registers demo champion/challenger models (MICROSCORE_AUTO_SEED=1).
CMD ["uvicorn", "microscore.api:app", "--host", "0.0.0.0", "--port", "8000"]
