#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE=""
REPETITIONS=1
METHODS="suql_baseline suql_v1 trummer_baseline trummer_v1"
CHEAP_MODEL="gemma4:e2b"
EXPENSIVE_MODEL="gemma4:26b"
OUTPUT_NAME=""
WALLTIME="24:00:00"
PARALLEL_WORKERS=4
PULL_MODELS=0
REQUIRE_GPU=1
NO_SEMANTIC_DICT=0
CALIBRATION_BUDGET="${CALIBRATION_BUDGET:-20}"
# Retain per-question run_metrics_repetitions.csv. Defaults on for multi-rep
# runs, since otherwise the per-repetition numbers are pruned after averaging.
KEEP_RUN_ARTIFACTS="${KEEP_RUN_ARTIFACTS:-}"
KRAKEN_GPU_MODEL="${KRAKEN_GPU_MODEL:-H200}"
AKER_HOST="${AKER_HOST:-remizova@aker.imag.fr}"
AKER_ROOT="${AKER_ROOT:-/home/daisy/remizova/lab_m2_benchmarks}"

usage() {
  echo "Usage: $0 --suite {10q|5q|3q|1q|scale_x<N>} [--repetitions N] [--methods '...'] [--cheap-model M] [--expensive-model M] [--output-name N] [--pull-models] [--allow-cpu] [--no-semantic-dict] [--calibration-budget N]"
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite) SUITE="$2"; shift 2;;
    --repetitions) REPETITIONS="$2"; shift 2;;
    --methods|--implementations) METHODS="$2"; shift 2;;
    --cheap-model) CHEAP_MODEL="$2"; shift 2;;
    --expensive-model|--main-model) EXPENSIVE_MODEL="$2"; shift 2;;
    --output-name) OUTPUT_NAME="$2"; shift 2;;
    --walltime) WALLTIME="$2"; shift 2;;
    --parallel-workers) PARALLEL_WORKERS="$2"; shift 2;;
    --pull-models) PULL_MODELS=1; shift;;
    --allow-cpu) REQUIRE_GPU=0; shift;;
    --no-semantic-dict) NO_SEMANTIC_DICT=1; shift;;
    --calibration-budget) CALIBRATION_BUDGET="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "ERROR: unknown option $1" >&2; usage >&2; exit 2;;
  esac
done
case "$SUITE" in 10q|5q|3q|1q|scale_x[0-9]*) ;; *) echo "ERROR: --suite must be 10q, 5q, 3q, 1q, or scale_x<N>" >&2; exit 2;; esac
[[ "$REPETITIONS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: repetitions must be positive" >&2; exit 2; }
valid=" suql_baseline suql_v1 trummer_baseline trummer_v1 "
for method in $METHODS; do [[ "$valid" == *" $method "* ]] || { echo "ERROR: unknown implementation $method" >&2; exit 2; }; done
STAMP="$(date +%Y%m%d_%H%M%S)"
[[ -n "$OUTPUT_NAME" ]] || OUTPUT_NAME="${SUITE}_${REPETITIONS}reps_${STAMP}"
# Multi-rep runs exist to measure spread, so keep the per-repetition metrics.
[[ -n "$KEEP_RUN_ARTIFACTS" ]] || { [[ "$REPETITIONS" -gt 1 ]] && KEEP_RUN_ARTIFACTS=1 || KEEP_RUN_ARTIFACTS=0; }
METHODS_SHELL="$(printf '%q' "$METHODS")"

AKER_HOST="$AKER_HOST" AKER_ROOT="$AKER_ROOT" bash "$HERE/sync_to_aker.sh"
ssh "$AKER_HOST" "
  set -Eeuo pipefail
  ROOT='$AKER_ROOT'; SUITE='$SUITE'; STAMP='$STAMP'
  JOBS=\"\$ROOT/benchmarks/\$SUITE/jobs\"; LOGS=\"\$ROOT/benchmarks/\$SUITE/logs\"
  mkdir -p \"\$JOBS\" \"\$LOGS\"
  WRAPPER=\"\$JOBS/run_\$STAMP.sh\"
  printf '%s\n' '#!/usr/bin/env bash' 'set -Eeuo pipefail' \
    'export AKER_ROOT=$AKER_ROOT' \
    'export SUITE=$SUITE' \
    'export REPETITIONS=$REPETITIONS' \
    'export METHODS=$METHODS_SHELL' \
    'export CHEAP_MODEL=$CHEAP_MODEL' \
    'export EXPENSIVE_MODEL=$EXPENSIVE_MODEL' \
    'export OUTPUT_NAME=$OUTPUT_NAME' \
    'export RUN_STAMP=$STAMP' \
    'export PARALLEL_WORKERS=$PARALLEL_WORKERS' \
    'export PULL_MODELS=$PULL_MODELS' \
    'export REQUIRE_GPU=$REQUIRE_GPU' \
    'export NO_SEMANTIC_DICT=$NO_SEMANTIC_DICT' \
    'export CALIBRATION_BUDGET=$CALIBRATION_BUDGET' \
    'export KEEP_RUN_ARTIFACTS=$KEEP_RUN_ARTIFACTS' \
    'exec bash $AKER_ROOT/benchmarks/shared/scripts/_aker_worker.sh' > \"\$WRAPPER\"
  chmod 700 \"\$WRAPPER\"
  if [[ '$AKER_HOST' == kraken* ]]; then
    oarsub --name \"benchmark-\$SUITE\" -l '/nodes=1/gpu=1,walltime=$WALLTIME' \
      -p \"gpumodel='$KRAKEN_GPU_MODEL'\" --project pr-daisyllm \
      -O \"\$LOGS/oar_%jobid%.out\" -E \"\$LOGS/oar_%jobid%.err\" \"\$WRAPPER\"
  else
    oarsub -n \"benchmark-\$SUITE\" -l '/nodes=1/gpu=1,walltime=$WALLTIME' \
      -O \"\$LOGS/oar_%jobid%.out\" -E \"\$LOGS/oar_%jobid%.err\" -S \"\$WRAPPER\"
  fi
"
echo "Output: $AKER_ROOT/benchmarks/$SUITE/outputs/$OUTPUT_NAME"
