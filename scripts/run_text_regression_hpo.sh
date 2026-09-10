#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

STUDY_DIR="${STUDY_DIR:-experiments/hpo_text_regression}"
mkdir -p "$STUDY_DIR"
exec 9>"$STUDY_DIR/.hpo.lock"
if ! flock -n 9; then
  echo "Another MentalBERT regression HPO process is already running." >&2
  exit 1
fi

uv run python scripts/tune_text_regression_hpo.py "$@"
