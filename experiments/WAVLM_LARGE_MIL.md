# Frozen WavLM-Large recording MIL experiment

This experiment tests whether learned recording-level attention improves over
fixed mean pooling across speech windows. It reuses the frozen WavLM-Large cache
and therefore never loads or fine-tunes the 316-million-parameter encoder.

Both downstream models use a fixed average of hidden states 16–21, concatenate
the cached within-window temporal standard deviation, and jointly predict:

- standardized continuous MADRS; and
- MADRS ≥20 with class-balanced binary loss.

The two models differ only in recording aggregation:

- `mean`: equal weight for every window after projection;
- `attention`: gated attention learns a normalized weight for every window.

Training batches contain recordings, not independently sampled windows. At most
32 randomly chosen windows from each recording are used in an epoch. Grouped CV
and final validation always use every window. Five subject-grouped folds choose
training duration, and the external validation set is excluded from early
stopping and classification-cutoff selection.

Inspect the command without training:

```bash
./scripts/run_wavlm_large_mil.sh --dry-run
```

Run on the GPU:

```bash
./scripts/run_wavlm_large_mil.sh
```

Results are written to `outputs/wavlm-large-mil/`. In addition to metrics and
predictions, `validation_attention_weights.csv` records the learned attention
assigned to every validation window for later failure-mode inspection.
