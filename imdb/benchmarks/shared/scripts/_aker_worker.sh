#!/usr/bin/env bash
# Generic OAR worker for any canonical benchmark suite.

set -Eeuo pipefail

AKER_ROOT="${AKER_ROOT:-/home/daisy/remizova/lab_m2_benchmarks}"
SUITE="${SUITE:?SUITE must be 10q, 5q, 3q, or 1q}"
BENCHMARK_ROOT="$AKER_ROOT/benchmarks"
SUITE_ROOT="$BENCHMARK_ROOT/$SUITE"
SHARED_ROOT="$BENCHMARK_ROOT/shared"
CHEAP_MODEL="${CHEAP_MODEL:-gemma4:e2b}"
EXPENSIVE_MODEL="${EXPENSIVE_MODEL:-gemma4:e4b}"
export PYTHONPATH="$AKER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export BENCHMARK_SUITE_ROOT="$SUITE_ROOT"
PULL_MODELS="${PULL_MODELS:-0}"
CASCADE_TARGET="${CASCADE_TARGET:-0.8}"
CALIBRATION_BUDGET="${CALIBRATION_BUDGET:-20}"
MANUAL_CONFIDENCE_THRESHOLD="${MANUAL_CONFIDENCE_THRESHOLD:-}"
CHEAP_BATCH_SIZE="${CHEAP_BATCH_SIZE:-8}"
EXPENSIVE_BATCH_SIZE="${EXPENSIVE_BATCH_SIZE:-32}"
MAX_EXPENSIVE_CALLS="${MAX_EXPENSIVE_CALLS:-4}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-4}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-3600}"
TOKEN_THRESHOLD="${TOKEN_THRESHOLD:-4096}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-512}"
MAX_MOVIE_BLOCK_SIZE="${MAX_MOVIE_BLOCK_SIZE:-25}"
MAX_REVIEW_BLOCK_SIZE="${MAX_REVIEW_BLOCK_SIZE:-8}"
REPETITIONS="${REPETITIONS:-1}"
METHODS="${METHODS:-suql_baseline suql_v1 trummer_baseline trummer_v1}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"
NO_SEMANTIC_DICT="${NO_SEMANTIC_DICT:-0}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_NAME="${OUTPUT_NAME:-${SUITE}_${REPETITIONS}reps_${RUN_STAMP}}"
OLLAMA_BIN="${OLLAMA_BIN:-}"

cd "$AKER_ROOT"
mkdir -p "$SUITE_ROOT/logs" "$SUITE_ROOT/outputs/$OUTPUT_NAME"

if [[ -z "${OAR_JOB_ID:-}" ]]; then
  echo "ERROR: this worker must run inside an OAR allocation." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: no NVIDIA GPU is visible. Refusing to run." >&2
  exit 1
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

if [[ ! -x "$AKER_ROOT/.venv/bin/python" ]]; then
  python3 -m venv "$AKER_ROOT/.venv"
fi
PYTHON="$AKER_ROOT/.venv/bin/python"
"$PYTHON" -m pip install -r "$BENCHMARK_ROOT/requirements.txt"

find_ollama() {
  if [[ -n "$OLLAMA_BIN" && -x "$OLLAMA_BIN" ]]; then printf '%s\n' "$OLLAMA_BIN"; return; fi
  if command -v ollama >/dev/null 2>&1; then command -v ollama; return; fi
  for candidate in "$HOME/.local/ollama/bin/ollama" "$HOME/.local/bin/ollama" "$HOME/bin/ollama" /usr/local/bin/ollama /usr/bin/ollama; do
    if [[ -x "$candidate" ]]; then printf '%s\n' "$candidate"; return; fi
  done
  return 1
}

OLLAMA_BIN="$(find_ollama || true)"
if [[ -z "$OLLAMA_BIN" ]]; then
  echo "ERROR: Ollama not found; set OLLAMA_BIN." >&2
  exit 1
fi

OLLAMA_PORT="${OLLAMA_PORT:-$((12000 + OAR_JOB_ID % 1000))}"
export OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"
API_BASE="http://$OLLAMA_HOST"
OLLAMA_LOG="$SUITE_ROOT/logs/ollama_${OAR_JOB_ID}_${RUN_STAMP}.log"
CONSOLE_LOG="$SUITE_ROOT/logs/benchmark_${OAR_JOB_ID}_${RUN_STAMP}.console.log"
ollama_pid=""
cleanup() {
  if [[ -n "$ollama_pid" ]] && kill -0 "$ollama_pid" >/dev/null 2>&1; then
    kill "$ollama_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

OLLAMA_HOST="$OLLAMA_HOST" OLLAMA_NUM_PARALLEL="$PARALLEL_WORKERS" \
  nohup "$OLLAMA_BIN" serve >"$OLLAMA_LOG" 2>&1 &
ollama_pid="$!"

ready=0
for _ in $(seq 1 90); do
  if "$PYTHON" - "$API_BASE" <<'PY'
import sys, urllib.request
try:
    urllib.request.urlopen(sys.argv[1] + "/api/tags", timeout=2)
except Exception:
    raise SystemExit(1)
PY
  then
    ready=1
    break
  fi
  sleep 2
done
if [[ "$ready" != "1" ]]; then
  echo "ERROR: Ollama did not become ready. See $OLLAMA_LOG" >&2
  exit 1
fi

for model in "$CHEAP_MODEL" "$EXPENSIVE_MODEL"; do
  model="${model#ollama/}"
  echo "Checking Ollama model: $model"
  if [[ "$PULL_MODELS" == "1" ]]; then
    OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" pull "$model"
  elif ! OLLAMA_HOST="$OLLAMA_HOST" "$OLLAMA_BIN" show "$model" >/dev/null 2>&1; then
    echo "ERROR: missing model '$model'; resubmit with PULL_MODELS=1." >&2
    exit 1
  fi
done

assert_ollama_gpu_resident() {
  local model="$1"
  local validation_timeout="${GPU_VALIDATION_TIMEOUT:-900}"
  echo "Verifying GPU residency for $model"
  "$PYTHON" - "$API_BASE" "$model" <<'PY' >/dev/null 2>&1 &
import json
import sys
import urllib.request

api_base, model = sys.argv[1], sys.argv[2]
payload = {
    "model": model,
    "prompt": "OK",
    "stream": False,
    "keep_alive": "30m",
    "options": {
        "num_predict": 1,
        "num_ctx": 2048,
        "temperature": 0,
    },
}
request = urllib.request.Request(
    api_base + "/api/generate",
    data=json.dumps(payload).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(request, timeout=900) as response:
    response.read()
PY
  local run_pid="$!"
  local seen_gpu=0
  local run_done=0
  for _ in $(seq 1 "$validation_timeout"); do
    if nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | grep -Ei 'ollama|llama' >/dev/null; then
      seen_gpu=1
      kill "$run_pid" >/dev/null 2>&1 || true
      wait "$run_pid" >/dev/null 2>&1 || true
      echo "Verified GPU residency for $model."
      return 0
    fi
    if ! kill -0 "$run_pid" >/dev/null 2>&1; then
      run_done=1
      break
    fi
    sleep 1
  done
  if [[ "$run_done" != "1" ]]; then
    echo "ERROR: GPU validation prompt for $model did not finish within ${validation_timeout}s." >&2
    echo "Last Ollama log lines from $OLLAMA_LOG:" >&2
    tail -n 80 "$OLLAMA_LOG" >&2 || true
    kill "$run_pid" >/dev/null 2>&1 || true
    wait "$run_pid" >/dev/null 2>&1 || true
    exit 1
  fi
  wait "$run_pid" || {
    echo "ERROR: GPU validation prompt failed for $model." >&2
    echo "Last Ollama log lines from $OLLAMA_LOG:" >&2
    tail -n 80 "$OLLAMA_LOG" >&2 || true
    exit 1
  }
  if [[ "$seen_gpu" != "1" ]]; then
    echo "ERROR: Ollama model '$model' did not appear in nvidia-smi compute apps." >&2
    echo "Last Ollama log lines from $OLLAMA_LOG:" >&2
    tail -n 80 "$OLLAMA_LOG" >&2 || true
    echo "Refusing to run because REQUIRE_GPU=$REQUIRE_GPU." >&2
    exit 1
  fi
}

if [[ "$REQUIRE_GPU" == "1" ]]; then
  assert_ollama_gpu_resident "${CHEAP_MODEL#ollama/}"
  assert_ollama_gpu_resident "${EXPENSIVE_MODEL#ollama/}"
fi

command=(
  "$PYTHON" -u "$SHARED_ROOT/scripts/run_all.py"
  --python "$PYTHON"
  --api-base "$API_BASE"
  --cheap-model "ollama/${CHEAP_MODEL#ollama/}"
  --expensive-model "ollama/${EXPENSIVE_MODEL#ollama/}"
  --cascade-target "$CASCADE_TARGET"
  --calibration-budget "$CALIBRATION_BUDGET"
  --cheap-batch-size "$CHEAP_BATCH_SIZE"
  --expensive-batch-size "$EXPENSIVE_BATCH_SIZE"
  --max-expensive-calls "$MAX_EXPENSIVE_CALLS"
  --request-timeout "$REQUEST_TIMEOUT"
  --token-threshold "$TOKEN_THRESHOLD"
  --max-completion-tokens "$MAX_COMPLETION_TOKENS"
  --max-movie-block-size "$MAX_MOVIE_BLOCK_SIZE"
  --max-review-block-size "$MAX_REVIEW_BLOCK_SIZE"
  --repetitions "$REPETITIONS"
  --methods $METHODS
  --output-dir "$SUITE_ROOT/outputs/$OUTPUT_NAME"
)
if [[ -n "$MANUAL_CONFIDENCE_THRESHOLD" ]]; then
  command+=(--manual-confidence-threshold "$MANUAL_CONFIDENCE_THRESHOLD")
fi
# Without this the per-question run_metrics_repetitions.csv files are pruned,
# which discards exactly the per-repetition numbers that seeded calibration
# exists to produce. Set KEEP_RUN_ARTIFACTS=1 whenever REPETITIONS > 1.
if [[ "${KEEP_RUN_ARTIFACTS:-0}" == "1" ]]; then
  command+=(--keep-run-artifacts)
fi
if [[ "$NO_SEMANTIC_DICT" == "1" ]]; then
  export SEMANTIC_DICT_PATH="/nonexistent/semantic_dict_disabled_for_ablation.json"
  export SEMANTIC_DICT_REQUIRED=0
fi

echo "Output directory: $SUITE_ROOT/outputs/$OUTPUT_NAME"
echo "Console log: $CONSOLE_LOG"
echo "Ollama log: $OLLAMA_LOG"
echo "Models: cheap=$CHEAP_MODEL expensive=$EXPENSIVE_MODEL"
echo "Methods: $METHODS"
echo "Suite: $SUITE"
echo "Semantic dict: $([[ "$NO_SEMANTIC_DICT" == "1" ]] && echo disabled || echo enabled)"
MPLBACKEND=Agg MPLCONFIGDIR="$SUITE_ROOT/.mplconfig" \
  "${command[@]}" 2>&1 | tee "$CONSOLE_LOG"

echo "Completed $SUITE."
echo "Outputs: $SUITE_ROOT/outputs/$OUTPUT_NAME"
echo "Console log: $CONSOLE_LOG"
