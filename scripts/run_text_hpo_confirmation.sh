#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/hpo_text_regression/confirmation}"
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
exec 9>"$OUTPUT_ROOT/.confirmation.lock"
if ! flock -n 9; then
  echo "Another text HPO confirmation suite is already running." >&2
  exit 1
fi

COMMON_ARGS=(
  --train-manifest /data/Clinical_vars/canbind_combined_20260806_train.csv
  --validation-manifest /data/Clinical_vars/canbind_combined_20260806_validation.csv
  --rdino-checkpoint assets/pretrained_rdino.pth
  --text-model mental/mental-bert-base-uncased
  --modality text
  --num-classes 4
  --eligibility-window-seconds 30
  --train-sampling recording
  --train-windows-per-recording 4
  --epochs 200
  --batch-size 8
  --workers 0
  --classification-weight 0.0
  --regression-weight 1.0
  --class-weighting none
  --standardize-regression-labels
  --lr-scheduler none
  --early-stopping-patience 5
  --validation-interval 5
)

run_experiment() {
  local source_trial="$1"
  local seed="$2"
  shift 2
  local name
  name="$(printf "trial-%04d-seed-%d" "$source_trial" "$seed")"
  local output_dir="$OUTPUT_ROOT/$name"
  local command=(
    uv run python -u train.py
    "${COMMON_ARGS[@]}"
    --seed "$seed"
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
  echo "Starting confirmation experiment: $name"
  "${command[@]}" 2>&1 | tee "$output_dir/training.log"
  uv run python plot_training.py --output-dir "$output_dir"
  touch "$output_dir/.complete"
}

for seed in 41 42 43; do
  run_experiment 36 "$seed" \
    --embedding-normalization batchnorm \
    --lora-rank 8 \
    --lora-alpha 8 \
    --gradient-accumulation-steps 1 \
    --learning-rate 1.1962402885307259e-5 \
    --weight-decay 1e-6 \
    --window-seconds 25 \
    --stride-seconds 10

  run_experiment 4 "$seed" \
    --embedding-normalization batchnorm \
    --lora-rank 4 \
    --lora-alpha 16 \
    --gradient-accumulation-steps 4 \
    --learning-rate 1.6358165301626213e-5 \
    --weight-decay 1e-4 \
    --window-seconds 20 \
    --stride-seconds 15
done

if (( DRY_RUN )); then
  echo "Dry run complete; no training was launched."
  exit 0
fi

uv run python scripts/summarize_text_hpo_confirmation.py \
  --output-root "$OUTPUT_ROOT" \
  --screening-root outputs/hpo_text_regression/screening

echo "Confirmation suite complete. See $OUTPUT_ROOT/report.md"
