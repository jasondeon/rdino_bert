#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

INPUT_DIR="${INPUT_DIR:-outputs/wavlm-large-layer-audit}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/wavlm-large-mil}"
DEVICE="${DEVICE:-cuda}"
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
  uv run python -u scripts/train_wavlm_large_mil.py
  --input-dir "$INPUT_DIR"
  --output-dir "$OUTPUT_DIR"
  --layers 16 17 18 19 20 21
  --madrs-threshold 20
  --include-temporal-std
  --projection-size 128
  --attention-size 64
  --dropout 0.25
  --regression-weight 1
  --classification-weight 1
  --learning-rate 3e-4
  --weight-decay 1e-4
  --batch-size 8
  --max-train-windows 32
  --cv-folds 5
  --seeds 40 41 42
  --max-epochs 250
  --patience 30
  --minimum-delta 1e-4
  --gradient-clip 1
  --bootstrap-samples 5000
  --device "$DEVICE"
)

if (( DRY_RUN )); then
  printf "Training command:\n  "
  printf "%q " "${command[@]}"
  printf "\n\nDry run complete; no training was launched.\n"
  exit 0
fi

if [[ ! -f "$INPUT_DIR/.complete" ]]; then
  echo "WavLM-Large cache is incomplete: $INPUT_DIR" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/.wavlm-mil.lock"
if ! flock -n 9; then
  echo "Another WavLM-Large MIL experiment is already running." >&2
  exit 1
fi
if [[ -f "$OUTPUT_DIR/.complete" ]]; then
  echo "MIL experiment already complete: $OUTPUT_DIR/report.md"
  exit 0
fi
if find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 \
  ! -name '.wavlm-mil.lock' -print -quit | grep -q .; then
  echo "Refusing to overwrite incomplete outputs in $OUTPUT_DIR" >&2
  echo "Move that directory aside, then rerun this script." >&2
  exit 1
fi

"${command[@]}" 2>&1 | tee "$OUTPUT_DIR/training.log"
touch "$OUTPUT_DIR/.complete"
echo "WavLM-Large MIL experiment complete. See $OUTPUT_DIR/report.md"
