#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/rdino-structural}"
RUN_NAME="06-balanced-window-preserved-gaps"
OUTPUT_DIR="$OUTPUT_ROOT/$RUN_NAME"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if (( $# > 0 )); then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

command=(
  uv run python -u train.py
  --train-manifest /data/Clinical_vars/canbind_combined_20260806_train.csv
  --validation-manifest /data/Clinical_vars/canbind_combined_20260806_validation.csv
  --rdino-yaml assets/rdino.yaml
  --rdino-checkpoint assets/pretrained_rdino.pth
  --output-dir "$OUTPUT_DIR"
  --modality audio
  --num-classes 4
  --embedding-normalization batchnorm
  --fusion-architecture two_layer
  --rdino-lora-scope terminal
  --lora-rank 2
  --lora-alpha 16
  --window-seconds 20
  --stride-seconds 15
  --eligibility-window-seconds 30
  --speaker-gap-policy preserve
  --train-sampling balanced_window
  --epochs 20
  --batch-size 8
  --gradient-accumulation-steps 2
  --workers 0
  --learning-rate 1e-4
  --weight-decay 1e-5
  --classification-weight 0.0
  --regression-weight 1.0
  --class-weighting none
  --standardize-regression-labels
  --lr-scheduler none
  --early-stopping-patience 3
  --validation-interval 1
  --seed 40
)

if (( DRY_RUN )); then
  printf "Run %s:\n  " "$RUN_NAME"
  printf "%q " "${command[@]}"
  printf "\n\nDry run complete; no training was launched.\n"
  exit 0
fi

mkdir -p "$OUTPUT_ROOT"
exec 9>"$OUTPUT_ROOT/.rdino-structural.lock"
if ! flock -n 9; then
  echo "Another RDINO experiment is already running." >&2
  exit 1
fi
if [[ -f "$OUTPUT_DIR/.complete" ]]; then
  echo "Experiment already complete: $OUTPUT_DIR"
  exit 0
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite incomplete experiment: $OUTPUT_DIR" >&2
  echo "Move that directory aside, then rerun this script." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
"${command[@]}" 2>&1 | tee "$OUTPUT_DIR/training.log"
uv run python plot_training.py --output-dir "$OUTPUT_DIR"
touch "$OUTPUT_DIR/.complete"
uv run python scripts/summarize_rdino_structural.py --output-root "$OUTPUT_ROOT"
echo "Final RDINO comparison complete. See $OUTPUT_ROOT/report.md"
