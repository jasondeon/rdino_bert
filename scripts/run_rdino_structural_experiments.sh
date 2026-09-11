#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/rdino-structural}"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if (( $# > 0 )); then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
exec 9>"$OUTPUT_ROOT/.rdino-structural.lock"
if ! flock -n 9; then
  echo "Another RDINO structural suite is already running." >&2
  exit 1
fi

COMMON_ARGS=(
  --train-manifest /data/Clinical_vars/canbind_combined_20260806_train.csv
  --validation-manifest /data/Clinical_vars/canbind_combined_20260806_validation.csv
  --rdino-checkpoint assets/pretrained_rdino.pth
  --modality audio
  --num-classes 4
  --embedding-normalization batchnorm
  --window-seconds 20
  --stride-seconds 15
  --eligibility-window-seconds 30
  --train-sampling recording
  --train-windows-per-recording 4
  --epochs 100
  --batch-size 8
  --gradient-accumulation-steps 2
  --workers 0
  --weight-decay 1e-5
  --classification-weight 0.0
  --regression-weight 1.0
  --class-weighting none
  --standardize-regression-labels
  --lr-scheduler none
  --early-stopping-patience 5
  --validation-interval 2
  --seed 40
)

run_experiment() {
  local name="$1"
  shift
  local output_dir="$OUTPUT_ROOT/$name"
  local command=(
    uv run python -u train.py
    "${COMMON_ARGS[@]}"
    --output-dir "$output_dir"
    "$@"
  )

  if (( DRY_RUN )); then
    printf "Run %s:\n  " "$name"
    printf "%q " "${command[@]}"
    printf "\n\n"
    return
  fi

  if [[ -f "$output_dir/.complete" ]]; then
    if [[ ! -f "$output_dir/training_curves.png" ]]; then
      uv run python plot_training.py --output-dir "$output_dir"
    fi
    echo "Skipping completed experiment: $name"
    return
  fi
  if [[ -e "$output_dir" ]]; then
    echo "Refusing to overwrite incomplete experiment: $output_dir" >&2
    echo "Move that directory aside, then rerun this script." >&2
    exit 1
  fi

  mkdir -p "$output_dir"
  echo
  echo "Starting RDINO structural experiment: $name"
  "${command[@]}" 2>&1 | tee "$output_dir/training.log"
  uv run python plot_training.py --output-dir "$output_dir"
  touch "$output_dir/.complete"
}

# Establish whether the old result was primarily caused by the very small LR.
run_experiment 00-corrected-terminal-two-layer \
  --fusion-architecture two_layer \
  --rdino-lora-scope terminal \
  --lora-rank 2 \
  --lora-alpha 16 \
  --learning-rate 1e-4

# Match the shallow 512 -> 50 audio head in the colleague's implementation.
run_experiment 01-shallow-terminal-uniform-lr \
  --fusion-architecture single_layer \
  --rdino-lora-scope terminal \
  --lora-rank 2 \
  --lora-alpha 16 \
  --learning-rate 1e-4

# Let the randomly initialized head learn faster than the pretrained adapter.
run_experiment 02-shallow-terminal-split-lr \
  --fusion-architecture single_layer \
  --rdino-lora-scope terminal \
  --lora-rank 2 \
  --lora-alpha 16 \
  --learning-rate 1e-4 \
  --audio-learning-rate 1e-4 \
  --head-learning-rate 1e-3

# Test whether adapting channel mixing throughout RDINO exposes paralinguistic signal.
run_experiment 03-shallow-all-pointwise-split-lr \
  --fusion-architecture single_layer \
  --rdino-lora-scope all_pointwise \
  --lora-rank 2 \
  --lora-alpha 16 \
  --learning-rate 1e-4 \
  --audio-learning-rate 1e-4 \
  --head-learning-rate 1e-3

# Reproduce the colleague-style frozen-embedding baseline with a fast head LR.
run_experiment 04-shallow-frozen-rdino \
  --fusion-architecture single_layer \
  --disable-rdino-lora \
  --learning-rate 2e-3 \
  --head-learning-rate 2e-3

if (( DRY_RUN )); then
  echo "Dry run complete; no training was launched."
  exit 0
fi

uv run python scripts/summarize_rdino_structural.py --output-root "$OUTPUT_ROOT"
echo "RDINO structural suite complete. See $OUTPUT_ROOT/report.md"
