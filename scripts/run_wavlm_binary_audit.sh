#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

INPUT_DIR="${INPUT_DIR:-outputs/wavlm-base-plus-layer-audit}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/wavlm-base-plus-binary-audit}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-5000}"
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
  uv run python -u scripts/analyze_wavlm_binary.py
  --input-dir "$INPUT_DIR"
  --output-dir "$OUTPUT_DIR"
  --thresholds 20 23
  --cv-folds 5
  --c-values 0.001 0.01 0.1 1 10
  --bootstrap-samples "$BOOTSTRAP_SAMPLES"
  --seed 40
  --max-iterations 3000
)

if (( DRY_RUN )); then
  printf "Analysis command:\n  "
  printf "%q " "${command[@]}"
  printf "\n\nDry run complete; no analysis was launched.\n"
  exit 0
fi

if [[ ! -f "$INPUT_DIR/.complete" ]]; then
  echo "WavLM layer cache is incomplete: $INPUT_DIR" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/.binary-audit.lock"
if ! flock -n 9; then
  echo "Another WavLM binary audit is already running." >&2
  exit 1
fi
if [[ -f "$OUTPUT_DIR/.complete" ]]; then
  echo "Binary audit already complete: $OUTPUT_DIR/report.md"
  exit 0
fi
if find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 \
  ! -name '.binary-audit.lock' -print -quit | grep -q .; then
  echo "Refusing to overwrite incomplete outputs in $OUTPUT_DIR" >&2
  echo "Move that directory aside, then rerun this script." >&2
  exit 1
fi

"${command[@]}" 2>&1 | tee "$OUTPUT_DIR/analysis.log"
touch "$OUTPUT_DIR/.complete"
echo "WavLM binary audit complete. See $OUTPUT_DIR/report.md"
