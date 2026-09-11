#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUTPUT_DIR="${OUTPUT_DIR:-outputs/wavlm-base-plus-layer-audit}"
MODEL_NAME="${MODEL_NAME:-microsoft/wavlm-base-plus}"
BATCH_SIZE="${BATCH_SIZE:-8}"
JOBS="${JOBS:--1}"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if (( $# > 0 )); then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

extract_command=(
  uv run python -u scripts/extract_wavlm_layers.py
  --train-manifest /data/Clinical_vars/canbind_combined_20260806_train.csv
  --validation-manifest /data/Clinical_vars/canbind_combined_20260806_validation.csv
  --output-dir "$OUTPUT_DIR"
  --model-name "$MODEL_NAME"
  --window-seconds 10
  --stride-seconds 7.5
  --eligibility-window-seconds 30
  --speaker-gap-policy preserve
  --batch-size "$BATCH_SIZE"
  --workers 0
  --cache-dtype float16
  --device auto
)
analysis_command=(
  uv run python -u scripts/analyze_wavlm_layers.py
  --input-dir "$OUTPUT_DIR"
  --cv-folds 5
  --bootstrap-samples 2000
  --seed 40
  --jobs "$JOBS"
)

if (( DRY_RUN )); then
  printf "Extraction command:\n  "
  printf "%q " "${extract_command[@]}"
  printf "\n\nAnalysis command:\n  "
  printf "%q " "${analysis_command[@]}"
  printf "\n\nDry run complete; no model was downloaded or GPU work launched.\n"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/.wavlm-layer-audit.lock"
if ! flock -n 9; then
  echo "Another WavLM layer audit is already running." >&2
  exit 1
fi
if [[ -f "$OUTPUT_DIR/.complete" ]]; then
  echo "Audit already complete: $OUTPUT_DIR/report.md"
  exit 0
fi
if find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 \
  ! -name '.wavlm-layer-audit.lock' -print -quit | grep -q .; then
  echo "Refusing to overwrite incomplete outputs in $OUTPUT_DIR" >&2
  echo "Move that directory aside, then rerun this script." >&2
  exit 1
fi

"${extract_command[@]}" 2>&1 | tee "$OUTPUT_DIR/extraction.log"
"${analysis_command[@]}" 2>&1 | tee "$OUTPUT_DIR/analysis.log"
touch "$OUTPUT_DIR/.complete"
echo "WavLM layer audit complete. See $OUTPUT_DIR/report.md"
