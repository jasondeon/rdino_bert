#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export TOKENIZERS_PARALLELISM=false
exec .venv/bin/python -u scripts/run_mentalbert_mixed.py "$@"
