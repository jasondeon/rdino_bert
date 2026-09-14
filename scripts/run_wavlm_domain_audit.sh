#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/wavlm-domain-audit}"
DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if (( $# > 0 )); then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

models=(base-plus large)
inputs=(outputs/wavlm-base-plus-layer-audit outputs/wavlm-large-layer-audit)

mkdir -p "$OUTPUT_ROOT"
exec 9>"$OUTPUT_ROOT/.domain-audit.lock"
if ! flock -n 9; then
  echo "Another WavLM domain audit is already running." >&2
  exit 1
fi

for index in "${!models[@]}"; do
  model="${models[$index]}"
  input_dir="${inputs[$index]}"
  output_dir="$OUTPUT_ROOT/$model"
  if [[ ! -f "$input_dir/extraction_config.json" ]]; then
    echo "Missing WavLM cache: $input_dir" >&2
    exit 1
  fi
  command=(
    uv run python -u scripts/analyze_wavlm_domain_generalization.py
    --input-dir "$input_dir"
    --output-dir "$output_dir"
    --madrs-threshold 20
    --ridge-alphas 100 1000 10000
    --c-values 0.001 0.01
    --diagnostic-folds 5
    --seed 40
  )
  if (( DRY_RUN )); then
    printf "%s command:\n  " "$model"
    printf "%q " "${command[@]}"
    printf "\n\n"
    continue
  fi
  if [[ -f "$output_dir/.complete" ]]; then
    echo "Already complete: $output_dir/report.md"
    continue
  fi
  if [[ -d "$output_dir" ]] && find "$output_dir" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing to overwrite incomplete output: $output_dir" >&2
    echo "Move it aside and rerun this script." >&2
    exit 1
  fi
  mkdir -p "$output_dir"
  "${command[@]}" 2>&1 | tee "$output_dir/analysis.log"
  touch "$output_dir/.complete"
done

if (( DRY_RUN )); then
  echo "Dry run complete; no cached embeddings were loaded."
else
  echo "Domain audits complete under $OUTPUT_ROOT"
fi
