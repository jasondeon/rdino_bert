#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

INPUT_DIR="${INPUT_DIR:-outputs/wavlm-base-plus-layer-audit}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/wavlm-base-plus-layer-mixer}"
DEVICE="${DEVICE:-auto}"
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
  uv run python -u scripts/train_wavlm_layer_mixer.py
  --input-dir "$INPUT_DIR"
  --output-dir "$OUTPUT_DIR"
  --folds 5
  --seeds 40 41 42
  --learning-rate 1e-3
  --weight-decay 1e-2
  --dropout 0.25
  --batch-size 32
  --max-epochs 500
  --patience 40
  --minimum-delta 1e-5
  --gradient-clip 1.0
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
  echo "WavLM layer audit is incomplete: $INPUT_DIR" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/.layer-mixer.lock"
if ! flock -n 9; then
  echo "Another WavLM layer mixer is running." >&2
  exit 1
fi
if [[ -f "$OUTPUT_DIR/.complete" ]]; then
  echo "Layer mixer already complete: $OUTPUT_DIR/report.md"
  exit 0
fi
if find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 \
  ! -name '.layer-mixer.lock' -print -quit | grep -q .; then
  echo "Refusing to overwrite incomplete outputs in $OUTPUT_DIR" >&2
  echo "Move that directory aside, then rerun this script." >&2
  exit 1
fi

"${command[@]}" 2>&1 | tee "$OUTPUT_DIR/training.log"
touch "$OUTPUT_DIR/.complete"
echo "WavLM layer mixer complete. See $OUTPUT_DIR/report.md"
