#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUTPUT_DIR="${OUTPUT_DIR:-outputs/opensmile-is09-manuscript-cohort}"
AUDIO_WORKERS="${AUDIO_WORKERS:-2}"
SMILE_WORKERS="${SMILE_WORKERS:-8}"
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

extract_command=(
  uv run python -u scripts/extract_opensmile_is09.py
  --train-manifest /data/Clinical_vars/rdino_bert_train_20260912.csv
  --validation-manifest /data/Clinical_vars/rdino_bert_test.csv
  --output-dir "$OUTPUT_DIR"
  --window-seconds 20
  --stride-seconds 15
  --eligibility-window-seconds 20
  --speaker-gap-policy preserve
  --batch-size 16
  --audio-workers "$AUDIO_WORKERS"
  --smile-workers "$SMILE_WORKERS"
)
analysis_command=(
  uv run python -u scripts/analyze_opensmile_is09.py
  --input-dir "$OUTPUT_DIR"
  --cv-folds 5
  --madrs-threshold 20
  --rf-search-iterations 40
  --rf-trees 1000
  --bootstrap-samples 5000
  --seed 40
  --jobs "$JOBS"
)

if (( DRY_RUN )); then
  printf "Extraction command:\n  "
  printf "%q " "${extract_command[@]}"
  printf "\n\nAnalysis command:\n  "
  printf "%q " "${analysis_command[@]}"
  printf "\n\nDry run complete; no extraction or fitting was launched.\n"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/.opensmile-is09.lock"
if ! flock -n 9; then
  echo "Another manuscript-cohort IS09 experiment is already running." >&2
  exit 1
fi
if [[ -f "$OUTPUT_DIR/.complete" ]]; then
  echo "Manuscript-cohort IS09 experiment already complete: $OUTPUT_DIR/report.md"
  exit 0
fi
if find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 \
  ! -name '.opensmile-is09.lock' -print -quit | grep -q .; then
  echo "Refusing to overwrite incomplete outputs in $OUTPUT_DIR" >&2
  echo "Move that directory aside, then rerun this script." >&2
  exit 1
fi

"${extract_command[@]}" 2>&1 | tee "$OUTPUT_DIR/extraction.log"
"${analysis_command[@]}" 2>&1 | tee "$OUTPUT_DIR/analysis.log"
touch "$OUTPUT_DIR/.complete"
echo "Manuscript-cohort IS09 baselines complete. See $OUTPUT_DIR/report.md"
