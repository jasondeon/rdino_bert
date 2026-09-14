#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

INPUT_DIR="${INPUT_DIR:-outputs/opensmile-is09-manuscript-cohort}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/opensmile-is09-longitudinal-change}"
JOBS="${JOBS:--1}"
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
  uv run python -u scripts/analyze_opensmile_longitudinal_change.py
  --input-dir "$INPUT_DIR"
  --output-dir "$OUTPUT_DIR"
  --folds 5
  --repeats 5
  --inner-folds 4
  --rf-trees 500
  --rf-min-leaf 5
  --bootstrap-samples 2000
  --jobs "$JOBS"
  --seed 40
)

if (( DRY_RUN )); then
  printf "Analysis command:\n  "
  printf "%q " "${command[@]}"
  printf "\n\nDry run complete; no cached features were loaded.\n"
  exit 0
fi

if [[ ! -f "$INPUT_DIR/extraction_config.json" || ! -f "$INPUT_DIR/train_features.npy" ]]; then
  echo "OpenSMILE cache is incomplete: $INPUT_DIR" >&2
  exit 1
fi
mkdir -p "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/.longitudinal-audit.lock"
if ! flock -n 9; then
  echo "Another longitudinal OpenSMILE audit is running." >&2
  exit 1
fi
if [[ -f "$OUTPUT_DIR/.complete" ]]; then
  echo "Audit already complete: $OUTPUT_DIR/report.md"
  exit 0
fi
if find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 \
  ! -name '.longitudinal-audit.lock' -print -quit | grep -q .; then
  echo "Refusing to overwrite incomplete output: $OUTPUT_DIR" >&2
  echo "Move that directory aside, then rerun this script." >&2
  exit 1
fi

"${command[@]}" 2>&1 | tee "$OUTPUT_DIR/analysis.log"
touch "$OUTPUT_DIR/.complete"
echo "Longitudinal OpenSMILE audit complete: $OUTPUT_DIR/report.md"
