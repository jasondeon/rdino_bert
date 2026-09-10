#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/text-recording-next}"
mkdir -p "$OUTPUT_ROOT"
exec 9>"$OUTPUT_ROOT/.suite.lock"
if ! flock -n 9; then
  echo "Another text recording experiment suite is already running." >&2
  exit 1
fi

COMMON_ARGS=(
  --train-manifest /data/Clinical_vars/canbind_combined_20260806_train.csv
  --validation-manifest /data/Clinical_vars/canbind_combined_20260806_validation.csv
  --rdino-checkpoint assets/pretrained_rdino.pth
  --text-model mental/mental-bert-base-uncased
  --modality text
  --embedding-normalization layernorm
  --lora-rank 2
  --lora-alpha 16
  --num-classes 4
  --window-seconds 30
  --stride-seconds 20
  --train-sampling recording
  --epochs 150
  --batch-size 8
  --workers 0
  --learning-rate 1e-5
  --weight-decay 1e-5
  --classification-weight 3.7841770535844463
  --regression-weight 1.0
  --class-weighting sqrt_inverse_frequency
  --standardize-regression-labels
  --lr-scheduler plateau
  --lr-scheduler-factor 0.7
  --lr-scheduler-patience 3
  --min-learning-rate 1e-7
  --early-stopping-patience 5
  --validation-interval 5
  --log-task-gradients
  --task-gradient-batches-per-epoch 3
)

run_experiment() {
  local name="$1"
  local seed="$2"
  local windows_per_recording="$3"
  local accumulation_steps="$4"
  local output_dir="$OUTPUT_ROOT/$name"

  if [[ -f "$output_dir/best_checkpoint.pt" && -f "$output_dir/training_history.csv" ]]; then
    echo "Skipping completed experiment: $name"
    return
  fi
  if [[ -e "$output_dir" ]]; then
    echo "Refusing to overwrite incomplete experiment: $output_dir" >&2
    echo "Move that directory aside, then run this script again." >&2
    exit 1
  fi

  mkdir -p "$output_dir"
  echo
  echo "Starting experiment: $name"
  uv run python -u train.py \
    "${COMMON_ARGS[@]}" \
    --seed "$seed" \
    --train-windows-per-recording "$windows_per_recording" \
    --gradient-accumulation-steps "$accumulation_steps" \
    --output-dir "$output_dir" 2>&1 | tee "$output_dir/training.log"
  uv run python plot_training.py --output-dir "$output_dir"
}

# Re-run seed 40 because weighted classification loss is now normalized by
# sample count, ensuring that class weights remain active in tiny batches.
run_experiment 00-k4-seed40 40 4 2
run_experiment 01-k4-seed41 41 4 2
run_experiment 02-k4-seed42 42 4 2

# Eight windows fill each physical batch with one recording. Accumulating four
# batches preserves four independent recordings per optimizer update.
run_experiment 03-k8-seed40 40 8 4

uv run python scripts/summarize_diagnostics.py --output-root "$OUTPUT_ROOT"
echo "Experiment suite complete. See $OUTPUT_ROOT/report.md"
