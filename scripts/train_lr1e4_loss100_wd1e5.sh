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
  --stride-jitter-seconds 3 \
  --epochs 100 \
  --batch-size 8 \
  --train-sampling window \
  --learning-rate 1e-4 \
  --classification-weight 1.0 \
  --weight-decay 1e-5 \
  --workers 0 \
  --early-stopping-patience 10 \
  --standardize-regression-labels \
  --augment-audio \
  --awgn-probability 0.5 \
  --awgn-snr-min-db 10 \
  --awgn-snr-max-db 30 \
  --reverb-probability 0.3 \
  --output-dir "outputs/$(basename "$0" .sh)"
