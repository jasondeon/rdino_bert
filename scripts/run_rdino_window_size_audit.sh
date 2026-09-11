#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/rdino-window-size-audit}"
REFERENCE_20_DIR="${REFERENCE_20_DIR:-outputs/rdino-embedding-audit-preserved-gaps}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-/data/Clinical_vars/canbind_combined_20260806_train.csv}"
VALIDATION_MANIFEST="${VALIDATION_MANIFEST:-/data/Clinical_vars/canbind_combined_20260806_validation.csv}"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if (( $# > 0 )); then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

# The completed 20-second corrected audit is the reference. Each new stride is
# 75% of its window length, matching the reference's 15/20 ratio.
WINDOW_SPECS=("5:3.75:window-05" "10:7.5:window-10" "30:22.5:window-30")

print_command() {
  printf "  "
  printf "%q " "$@"
  printf "\n"
}

extraction_command() {
  local window="$1"
  local stride="$2"
  local output_dir="$3"
  printf '%s\0' \
    uv run python -u scripts/extract_rdino_embeddings.py \
    --train-manifest "$TRAIN_MANIFEST" \
    --validation-manifest "$VALIDATION_MANIFEST" \
    --rdino-yaml assets/rdino.yaml \
    --rdino-checkpoint assets/pretrained_rdino.pth \
    --output-dir "$output_dir" \
    --window-seconds "$window" \
    --stride-seconds "$stride" \
    --eligibility-window-seconds 30 \
    --speaker-gap-policy preserve \
    --batch-size 8 \
    --workers 0 \
    --device auto
}

if (( DRY_RUN )); then
  echo "20-second reference (must be complete before a real run):"
  echo "  $REFERENCE_20_DIR"
  for spec in "${WINDOW_SPECS[@]}"; do
    IFS=: read -r window stride name <<<"$spec"
    output_dir="$OUTPUT_ROOT/$name"
    mapfile -d '' -t command < <(extraction_command "$window" "$stride" "$output_dir")
    printf "\n%s-second extraction:\n" "$window"
    print_command "${command[@]}"
    echo "Analysis:"
    print_command uv run python scripts/analyze_rdino_embeddings.py \
      --input-dir "$output_dir" --cv-folds 5 --bootstrap-samples 2000 --seed 40
  done
  printf "\nFinal comparison:\n"
  print_command uv run python scripts/summarize_rdino_window_audit.py \
    --output-root "$OUTPUT_ROOT" --reference-20-dir "$REFERENCE_20_DIR" \
    --bootstrap-samples 2000 --seed 40
  echo "Dry run complete; no extraction was launched."
  exit 0
fi

if [[ ! -f "$REFERENCE_20_DIR/.complete" ]]; then
  echo "The corrected 20-second reference is not complete: $REFERENCE_20_DIR" >&2
  echo "Finish ./scripts/run_rdino_embedding_audit.sh before running this suite." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
exec 9>"$OUTPUT_ROOT/.window-size-audit.lock"
if ! flock -n 9; then
  echo "Another RDINO window-size audit is already running." >&2
  exit 1
fi
if [[ -f "$OUTPUT_ROOT/.complete" ]]; then
  echo "Window-size audit already complete: $OUTPUT_ROOT/report.md"
  exit 0
fi

for spec in "${WINDOW_SPECS[@]}"; do
  IFS=: read -r window stride name <<<"$spec"
  output_dir="$OUTPUT_ROOT/$name"
  if [[ -f "$output_dir/.complete" ]]; then
    echo "Skipping completed ${window}-second audit: $output_dir"
    continue
  fi
  if [[ -d "$output_dir" ]] && find "$output_dir" -mindepth 1 -maxdepth 1 \
    -print -quit | grep -q .; then
    echo "Refusing to overwrite incomplete audit outputs in $output_dir" >&2
    echo "Move that directory aside, then rerun this script." >&2
    exit 1
  fi
  mkdir -p "$output_dir"
  mapfile -d '' -t command < <(extraction_command "$window" "$stride" "$output_dir")
  echo "Starting ${window}-second frozen RDINO embedding extraction."
  "${command[@]}" 2>&1 | tee "$output_dir/extraction.log"
  uv run python scripts/analyze_rdino_embeddings.py \
    --input-dir "$output_dir" \
    --cv-folds 5 \
    --bootstrap-samples 2000 \
    --seed 40 2>&1 | tee "$output_dir/analysis.log"
  touch "$output_dir/.complete"
done

uv run python scripts/summarize_rdino_window_audit.py \
  --output-root "$OUTPUT_ROOT" \
  --reference-20-dir "$REFERENCE_20_DIR" \
  --bootstrap-samples 2000 \
  --seed 40 2>&1 | tee "$OUTPUT_ROOT/summary.log"
touch "$OUTPUT_ROOT/.complete"
echo "RDINO window-size audit complete. See $OUTPUT_ROOT/report.md"
