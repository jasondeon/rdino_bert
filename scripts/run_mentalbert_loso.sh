#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# The Python runner holds a process lock, checkpoints each epoch, and resumes
# only when bundle, model revision, code, and experiment settings still match.
export TOKENIZERS_PARALLELISM=false
exec .venv/bin/python -u scripts/run_mentalbert_loso.py "$@"
