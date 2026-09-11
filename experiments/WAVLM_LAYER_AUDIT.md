# Frozen WavLM Base+ layer audit

## Purpose

Determine whether depression-regression signal exists in a particular WavLM depth
before building a trainable WavLM branch. This is a representation audit, not a
hyperparameter search.

## Design

- `microsoft/wavlm-base-plus` remains completely frozen.
- Audio uses the corrected primary-speaker preprocessing: natural unlabeled gaps
  are preserved and detected non-primary speakers form hard window boundaries.
- Ten-second windows and 7.5-second stride follow the scale used by recent
  layer-hierarchical depression work.
- A common 30-second eligibility requirement preserves the same recording cohort
  used in the RDINO window audit.
- Hidden-state index 0 is WavLM's projected convolutional representation. Indices
  1–12 are the outputs of transformer blocks 1–12.
- Each window stores the temporal mean and temporal standard deviation from every
  hidden state. Float16 is used only for the on-disk cache; pooling and probes use
  float64.
- For every layer, recording-level ridge probes compare temporal mean alone against
  temporal mean plus temporal standard deviation.
- Ridge regularization and the single reported selected configuration are chosen by
  subject-grouped cross-validation on training recordings. Validation results for
  all other layers are exploratory diagnostics.

## Run

Inspect the commands without downloading the model or using the GPU:

```bash
./scripts/run_wavlm_layer_audit.sh --dry-run
```

Run independently of Codex:

```bash
chmod +x scripts/run_wavlm_layer_audit.sh
./scripts/run_wavlm_layer_audit.sh
```

If memory is limited, reduce the extraction batch size:

```bash
BATCH_SIZE=4 ./scripts/run_wavlm_layer_audit.sh
```

The first real run downloads WavLM Base+ from Hugging Face. Results are written to
`outputs/wavlm-base-plus-layer-audit/`:

- `report.md`: the training-CV-selected result and all exploratory layer results;
- `layer_probe_curves.png`: CV RMSE, validation R²/RMSE, and prediction spread;
- `layer_probe_summary.csv`: machine-readable metrics;
- `selected_probe_predictions.csv`: recording-level predictions from the
  training-CV-selected configuration.

The two cached statistics arrays require roughly 1.7 GB at the default settings.
An interrupted extraction is never silently reused; move an incomplete output
directory aside before restarting.
