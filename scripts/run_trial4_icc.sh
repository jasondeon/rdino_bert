#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SEED="${SEED:-44}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/trial4-icc-seed-${SEED}}"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if (( $# > 0 )); then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

ARGS=(
  --train-manifest /data/Clinical_vars/canbind_combined_20260806_train.csv
  --validation-manifest /data/Clinical_vars/canbind_combined_20260806_validation.csv
  --rdino-checkpoint assets/pretrained_rdino.pth
  --output-dir "$OUTPUT_DIR"
  --text-model mental/mental-bert-base-uncased
  --modality text
  --embedding-normalization batchnorm
  --lora-rank 4
  --lora-alpha 16
  --num-classes 4
  --window-seconds 20
  --stride-seconds 15
  --eligibility-window-seconds 30
  --train-sampling recording
  --train-windows-per-recording 4
  --epochs 200
  --batch-size 8
  --gradient-accumulation-steps 4
  --workers 0
  --learning-rate 1.6358165301626213e-5
  --weight-decay 1e-4
  --classification-weight 0.0
  --regression-weight 1.0
  --class-weighting none
  --standardize-regression-labels
  --lr-scheduler none
  --early-stopping-patience 5
  --validation-interval 5
  --seed "$SEED"
)
COMMAND=(uv run python -u train.py "${ARGS[@]}")

if (( DRY_RUN )); then
  printf "Trial 4 ICC run:\n  "
  printf "%q " "${COMMAND[@]}"
  printf "\n"
  exit 0
fi

if [[ -f "$OUTPUT_DIR/.complete" ]]; then
  echo "Run already complete: $OUTPUT_DIR"
  exit 0
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite incomplete run: $OUTPUT_DIR" >&2
  echo "Move that directory aside, then rerun this script." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/.training.lock"
if ! flock -n 9; then
  echo "Another process is already using: $OUTPUT_DIR" >&2
  exit 1
fi

"${COMMAND[@]}" 2>&1 | tee "$OUTPUT_DIR/training.log"
uv run python plot_training.py --output-dir "$OUTPUT_DIR"
touch "$OUTPUT_DIR/.complete"

echo "Trial 4 ICC run complete: $OUTPUT_DIR"
