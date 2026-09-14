#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

INPUT_DIR="${INPUT_DIR:-outputs/wavlm-large-layer-audit}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/wavlm-large-longitudinal-selection-control}"
SHUFFLES="${SHUFFLES:-100}"
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
  uv run python -u scripts/analyze_wavlm_longitudinal_selection_control.py
  --input-dir "$INPUT_DIR"
  --output-dir "$OUTPUT_DIR"
  --layers 0 4 8 12 16 20 24
  --poolings mean mean_temporal_std
  --ridge-alphas 100 1000 10000 100000
  --inner-site-folds 4
  --shuffles "$SHUFFLES"
  --bootstrap-samples 5000
  --seed 42
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
exec 9>"$OUTPUT_DIR/.selection-control.lock"
if ! flock -n 9; then
  echo "Another selection-aware longitudinal control is running." >&2
  exit 1
fi
if [[ -f "$OUTPUT_DIR/.complete" ]]; then
  echo "Selection-aware control already complete: $OUTPUT_DIR/report.md"
  exit 0
fi

"${command[@]}" 2>&1 | tee -a "$OUTPUT_DIR/analysis.log"
touch "$OUTPUT_DIR/.complete"
echo "Selection-aware longitudinal control complete: $OUTPUT_DIR/report.md"
