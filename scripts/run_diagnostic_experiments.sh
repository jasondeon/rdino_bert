#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/diagnostic-suite}"
mkdir -p "$OUTPUT_ROOT"
exec 9>"$OUTPUT_ROOT/.diagnostic-suite.lock"
if ! flock -n 9; then
  echo "Another diagnostic experiment suite is already running." >&2
  exit 1
fi

COMMON_ARGS=(
  --train-manifest /data/Clinical_vars/canbind_combined_20260806_train.csv
  --validation-manifest /data/Clinical_vars/canbind_combined_20260806_validation.csv
  --num-classes 4
  --rdino-checkpoint assets/pretrained_rdino.pth
  --window-seconds 30
  --stride-seconds 20
  --epochs 100
  --batch-size 8
  --workers 0
  --learning-rate 3.218950677646652e-6
  --weight-decay 1e-5
  --classification-weight 3.7841770535844463
  --regression-weight 1.0
  --class-weighting sqrt_inverse_frequency
  --lr-scheduler plateau
  --lr-scheduler-factor 0.7
  --lr-scheduler-patience 3
  --min-learning-rate 1e-7
  --early-stopping-patience 5
  --standardize-regression-labels
  --log-task-gradients
  --task-gradient-batches-per-epoch 10
  --seed 40
)

run_experiment() {
  local name="$1"
  shift
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
    --output-dir "$output_dir" \
    "$@" 2>&1 | tee "$output_dir/training.log"
  uv run python plot_training.py --output-dir "$output_dir"
}

# Re-establish the previous HPO configuration with only RDINO BatchNorm statistics
# frozen, and collect task-gradient diagnostics.
run_experiment 00-frozen-rdino-bn-reference \
  --train-sampling window \
  --embedding-normalization batchnorm

# New corrected baseline: equal recording contribution, window-level losses, and
# batch-independent normalization of the modality embeddings.
run_experiment 01-balanced-layernorm-multimodal \
  --train-sampling balanced_window \
  --embedding-normalization layernorm

run_experiment 02-text-only \
  --train-sampling balanced_window \
  --embedding-normalization layernorm \
  --modality text

run_experiment 03-audio-only \
  --train-sampling balanced_window \
  --embedding-normalization layernorm \
  --modality audio

run_experiment 04-no-lora \
  --train-sampling balanced_window \
  --embedding-normalization layernorm \
  --disable-text-lora \
  --disable-rdino-lora

run_experiment 05-plain-bert \
  --train-sampling balanced_window \
  --embedding-normalization layernorm \
  --text-model bert-base-uncased

uv run python scripts/summarize_diagnostics.py --output-root "$OUTPUT_ROOT"
echo "Diagnostic suite complete. See $OUTPUT_ROOT/report.md"
