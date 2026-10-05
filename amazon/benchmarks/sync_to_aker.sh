#!/usr/bin/env bash
set -Eeuo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAB_ROOT="$(cd "$HERE/.." && pwd)"
AKER_HOST="${AKER_HOST:-remizova@aker.imag.fr}"
AKER_ROOT="${AKER_ROOT:-/home/daisy/remizova/lab_m2_benchmarks}"

IMPLEMENTATIONS=(
  "approaches/project SUQL/baseline"
  "approaches/project SUQL/v1"
  "approaches/project Trummer/baseline"
  "approaches/project Trummer/v1"
)
for relative in "${IMPLEMENTATIONS[@]}"; do
  [[ -d "$LAB_ROOT/$relative" ]] || { echo "ERROR: missing $LAB_ROOT/$relative" >&2; exit 1; }
done

ssh "$AKER_HOST" "mkdir -p '$AKER_ROOT/benchmarks' '$AKER_ROOT/project SUQL' '$AKER_ROOT/project Trummer' '$AKER_ROOT/data' '$AKER_ROOT/semantic_dict'"
# Remove implementation directories retired by the canonical four-method layout.
ssh "$AKER_HOST" "rm -rf \
  '$AKER_ROOT/project SUQL/Stage_1' \
  '$AKER_ROOT/project SUQL/Stage_2' \
  '$AKER_ROOT/project SUQL/src_baseline' \
  '$AKER_ROOT/project SUQL/src_baseline_stage1' \
  '$AKER_ROOT/project SUQL/src_baseline_stage2' \
  '$AKER_ROOT/project Trummer/heterogen_v1' \
  '$AKER_ROOT/project Trummer/heterogen_v2' \
  '$AKER_ROOT/project Trummer/heterogen_v2_2' \
  '$AKER_ROOT/project Trummer/heterogen_v2_3' \
  '$AKER_ROOT/project Trummer/heterogen_v3' \
  '$AKER_ROOT/project Trummer/heterogen_v3_2'"
remote_path() { printf "%s:%q" "$AKER_HOST" "$1"; }
# multi_model_experiments holds results pulled back down from the cluster;
# pushing them up again is pure waste. .venv matters because semantic_dict/
# carries a ~900MB virtualenv that times the transfer out.
rsync -av --delete --exclude outputs/ --exclude logs/ --exclude jobs/ --exclude probes/ \
  --exclude .mplconfig/ --exclude __pycache__/ --exclude .venv/ \
  --exclude multi_model_experiments/ \
  "$HERE/" "$(remote_path "$AKER_ROOT/benchmarks/")"
rsync -av --delete --exclude __pycache__/ --exclude sources/ --exclude .venv/ \
  "$LAB_ROOT/data/" "$(remote_path "$AKER_ROOT/data/")"
for relative in "${IMPLEMENTATIONS[@]}"; do
  ssh "$AKER_HOST" "mkdir -p '$AKER_ROOT/$relative'"
  rsync -av --delete --exclude outputs/ --exclude benchmarks/ --exclude model_sweeps/ \
    --exclude .mplconfig/ --exclude __pycache__/ \
    "$LAB_ROOT/$relative/" "$(remote_path "$AKER_ROOT/$relative/")"
done
ssh "$AKER_HOST" "chmod +x '$AKER_ROOT/benchmarks/shared/scripts/_aker_worker.sh'"
echo "Synced canonical benchmarks and all amazon-compatible implementations to $AKER_HOST:$AKER_ROOT"
