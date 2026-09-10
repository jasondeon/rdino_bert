#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUTPUT_DIR="${OUTPUT_DIR:-outputs/text-regression-only-short-seed40}"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to overwrite existing output directory: $OUTPUT_DIR" >&2
  echo "Set OUTPUT_DIR to a new path or move the existing directory aside." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/.training.lock"
if ! flock -n 9; then
  echo "Another process is already using: $OUTPUT_DIR" >&2
  exit 1
fi

ARGS=(
  --train-manifest /data/Clinical_vars/canbind_combined_20260806_train.csv
  --validation-manifest /data/Clinical_vars/canbind_combined_20260806_validation.csv
  --rdino-checkpoint assets/pretrained_rdino.pth
  --output-dir "$OUTPUT_DIR"
  --text-model mental/mental-bert-base-uncased
  --modality text
  --embedding-normalization layernorm
  --lora-rank 2
  --lora-alpha 16
  --num-classes 4
  --window-seconds 30
  --stride-seconds 20
  --train-sampling recording
  --train-windows-per-recording 4
  --epochs 100
  --batch-size 8
  --gradient-accumulation-steps 2
  --workers 0
  --learning-rate 1e-5
  --weight-decay 1e-5
  --classification-weight 0.0
  --regression-weight 1.0
  --class-weighting sqrt_inverse_frequency
  --standardize-regression-labels
  --lr-scheduler plateau
  --lr-scheduler-factor 0.7
  --lr-scheduler-patience 3
  --min-learning-rate 1e-7
  --early-stopping-patience 5
  --validation-interval 5
  --seed 40
)

printf "Starting MentalBERT regression-only control:\n  "
printf "%q " uv run python -u train.py "${ARGS[@]}"
printf "\n\n"

uv run python -u train.py "${ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/training.log"
uv run python plot_training.py --output-dir "$OUTPUT_DIR"

echo "Experiment complete: $OUTPUT_DIR"
