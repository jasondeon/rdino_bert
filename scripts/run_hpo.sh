#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

mkdir -p experiments/hpo_existing_settings
exec 9>experiments/hpo_existing_settings/.hpo.lock
if ! flock -n 9; then
  echo "Another hyperparameter search is already running." >&2
  exit 1
fi

uv run python scripts/tune_hyperparameters.py "$@"
