#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUTPUT_DIR="${OUTPUT_DIR:-outputs/rdino-embedding-audit-preserved-gaps}"
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
  uv run python -u scripts/extract_rdino_embeddings.py
  --train-manifest /data/Clinical_vars/canbind_combined_20260806_train.csv
  --validation-manifest /data/Clinical_vars/canbind_combined_20260806_validation.csv
  --rdino-yaml assets/rdino.yaml
  --rdino-checkpoint assets/pretrained_rdino.pth
  --output-dir "$OUTPUT_DIR"
  --window-seconds 20
  --stride-seconds 15
  --eligibility-window-seconds 30
  --speaker-gap-policy preserve
  --batch-size 8
  --workers 0
  --device auto
)

if (( DRY_RUN )); then
  printf "Extraction command:\n  "
  printf "%q " "${command[@]}"
  printf "\n\nAnalysis command:\n  "
  printf "%q " uv run python scripts/analyze_rdino_embeddings.py \
    --input-dir "$OUTPUT_DIR" --cv-folds 5 --bootstrap-samples 2000 --seed 40
  printf "\n\nDry run complete; no extraction was launched.\n"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
exec 9>"$OUTPUT_DIR/.embedding-audit.lock"
if ! flock -n 9; then
  echo "Another RDINO embedding audit is already running." >&2
  exit 1
fi
if [[ -f "$OUTPUT_DIR/.complete" ]]; then
  echo "Audit already complete: $OUTPUT_DIR/report.md"
  exit 0
fi
if find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 \
  ! -name '.embedding-audit.lock' -print -quit | grep -q .; then
  echo "Refusing to overwrite incomplete audit outputs in $OUTPUT_DIR" >&2
  echo "Move that directory aside, then rerun this script." >&2
  exit 1
fi

"${command[@]}" 2>&1 | tee "$OUTPUT_DIR/extraction.log"
uv run python scripts/analyze_rdino_embeddings.py \
  --input-dir "$OUTPUT_DIR" \
  --cv-folds 5 \
  --bootstrap-samples 2000 \
  --seed 40 2>&1 | tee "$OUTPUT_DIR/analysis.log"
touch "$OUTPUT_DIR/.complete"
echo "RDINO embedding audit complete. See $OUTPUT_DIR/report.md"
