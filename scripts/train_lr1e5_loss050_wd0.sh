#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv run python train.py \
  --train-manifest /data/Clinical_vars/canbind_combined_20260806_train.csv \
  --validation-manifest /data/Clinical_vars/canbind_combined_20260806_validation.csv \
  --num-classes 4 \
  --rdino-checkpoint assets/pretrained_rdino.pth \
  --window-seconds 30 \
  --stride-seconds 20 \
  --epochs 50 \
  --batch-size 8 \
  --train-sampling window \
  --learning-rate 1e-5 \
  --classification-weight 0.5 \
  --weight-decay 0 \
  --workers 0 \
  --early-stopping-patience 5 \
  --standardize-regression-labels \
  --output-dir "outputs/$(basename "$0" .sh)"
