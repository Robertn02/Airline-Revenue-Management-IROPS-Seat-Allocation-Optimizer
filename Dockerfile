# Multi-stage Dockerfile for Reroute API
# Stage 1: build the trained model during image build (uses more memory)
# Stage 2: minimal runtime image with the pre-built model

# ============================================================
# BUILD STAGE
# ============================================================
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies for LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY pyproject.toml README.md ./
COPY reroute/ ./reroute/

# Install in editable mode with API extras
RUN pip install --no-cache-dir -e ".[api]"

# Pre-train the risk model so we don't train at runtime
RUN python -c "from reroute.model.risk import train_from_scenarios; \
               from reroute.sim.generator import generate_dataset; \
               import os; os.makedirs('results', exist_ok=True); \
               scns = generate_dataset(n_scenarios=200, seed=42); \
               m, _, _ = train_from_scenarios(scns); \
               m.save('results/model.pkl'); \
               print('Model trained, AUC:', m.train_results.auc)"

# Pre-generate demo scenarios
RUN python -c "from reroute.cli.export_demo import run_export; \
               run_export(n_scenarios=12, model_path='results/model.pkl', \
                          output_path='results/scenarios_for_demo.json')"

# ============================================================
# RUNTIME STAGE
# ============================================================
FROM python:3.12-slim AS runtime

WORKDIR /app

# Install runtime libs only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/* \
 && apt-get clean

# Copy installed packages from build stage
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code and pre-built artifacts
COPY --from=builder /build/reroute /app/reroute
COPY --from=builder /build/results /app/results
COPY --from=builder /build/pyproject.toml /app/

# Render injects the PORT env var
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Use shell form to expand $PORT at runtime
CMD uvicorn reroute.api.server:create_app --host 0.0.0.0 --port $PORT --factory --workers 1
