#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

INPUT_DIR="${INPUT_DIR:-outputs/wavlm-large-layer-audit}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/wavlm-large-longitudinal-pairing-control}"
SHUFFLES="${SHUFFLES:-500}"
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
  uv run python -u scripts/analyze_wavlm_longitudinal_pairing_control.py
  --input-dir "$INPUT_DIR"
  --output-dir "$OUTPUT_DIR"
  --layer 0
  --ridge-alpha 1000
  --shuffles "$SHUFFLES"
  --seed 41
)

if (( DRY_RUN )); then
  printf "Analysis command:\n  "
  printf "%q " "${command[@]}"
  printf "\n\nDry run complete; no cached embeddings were loaded.\n"
  exit 0
fi

if [[ ! -f "$INPUT_DIR/extraction_config.json" || ! -f "$INPUT_DIR/train_layer_means.npy" ]]; then
  echo "WavLM cache is incomplete: $INPUT_DIR" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/.pairing-control.lock"
if ! flock -n 9; then
  echo "Another longitudinal pairing control is running." >&2
  exit 1
fi
if [[ -f "$OUTPUT_DIR/.complete" ]]; then
  echo "Pairing control already complete: $OUTPUT_DIR/report.md"
  exit 0
fi
if find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 \
  ! -name '.pairing-control.lock' -print -quit | grep -q .; then
  echo "Refusing to overwrite incomplete output: $OUTPUT_DIR" >&2
  echo "Move that directory aside, then rerun this script." >&2
  exit 1
fi

"${command[@]}" 2>&1 | tee "$OUTPUT_DIR/analysis.log"
touch "$OUTPUT_DIR/.complete"
echo "Longitudinal pairing control complete: $OUTPUT_DIR/report.md"
