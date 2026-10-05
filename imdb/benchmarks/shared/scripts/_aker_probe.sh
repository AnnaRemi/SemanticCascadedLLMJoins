#!/usr/bin/env bash
# OAR worker: per-model latency / log-prob / signal probe (see latency_probe.py).
# Env: AKER_ROOT, PROBE_MODELS (comma list), PROBE_QUESTIONS, PROBE_NAME.

set -Eeuo pipefail

AKER_ROOT="${AKER_ROOT:?AKER_ROOT required}"
PROBE_MODELS="${PROBE_MODELS:?PROBE_MODELS required}"
PROBE_QUESTIONS="${PROBE_QUESTIONS:-q_01,q_02,q_05}"
PROBE_NAME="${PROBE_NAME:-probe_${OAR_JOB_ID:-local}}"
BENCHMARK_ROOT="$AKER_ROOT/benchmarks"
OUT_DIR="$BENCHMARK_ROOT/probes/$PROBE_NAME"
PYTHON="$AKER_ROOT/.venv/bin/python"
OLLAMA_BIN="${OLLAMA_BIN:-$HOME/.local/ollama/bin/ollama}"
mkdir -p "$OUT_DIR"

if [[ -z "${OAR_JOB_ID:-}" ]] || ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: needs an OAR GPU allocation." >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee "$OUT_DIR/gpu.txt"
"$PYTHON" -c "import httpx, numpy, pandas, scipy" # fail fast if the venv is incomplete

export OLLAMA_HOST="127.0.0.1:$((12000 + OAR_JOB_ID % 1000))"
API_BASE="http://$OLLAMA_HOST"
"$OLLAMA_BIN" --version >"$OUT_DIR/ollama_version.txt" 2>&1 || true
# One request at a time: this probe measures single-stream latency.
OLLAMA_NUM_PARALLEL=1 nohup "$OLLAMA_BIN" serve >"$OUT_DIR/ollama.log" 2>&1 &
ollama_pid=$!
trap 'kill $ollama_pid >/dev/null 2>&1 || true' EXIT
for _ in $(seq 1 90); do
  "$PYTHON" -c "import urllib.request; urllib.request.urlopen('$API_BASE/api/tags', timeout=2)" 2>/dev/null && break
  sleep 2
done

export PYTHONPATH="$AKER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -u "$BENCHMARK_ROOT/shared/scripts/latency_probe.py" \
  --api-base "$API_BASE" \
  --suql-dir "$AKER_ROOT/approaches/project SUQL/v1" \
  --data "$BENCHMARK_ROOT/shared/probe_data.json" \
  --out "$OUT_DIR/probe.json" \
  --models "$PROBE_MODELS" \
  --questions "$PROBE_QUESTIONS" 2>&1 | tee "$OUT_DIR/probe.console.log"
echo "Probe output: $OUT_DIR/probe.json"
