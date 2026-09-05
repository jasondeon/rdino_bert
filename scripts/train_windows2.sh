#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv run python train.py \
  --train-manifest /data/Clinical_vars/canbind_combined_20260806_train.csv \
  --validation-manifest /data/Clinical_vars/canbind_combined_20260806_validation.csv \
  --num-classes 4 \
  --rdino-checkpoint assets/pretrained_rdino.pth \
  --window-seconds 20 \
  --stride-seconds 10 \
  --epochs 50 \
  --batch-size 8 \
  --train-windows-per-recording 4 \
  --workers 0 \
  --early-stopping-patience 5 \
  --standardize-regression-labels \
  --output-dir "outputs/$(basename "$0" .sh)"
