# WavLM binary MADRS signal audit

This experiment reuses the frozen WavLM Base+ window cache and tests two binary
severity definitions: MADRS ≥20 and MADRS ≥23.

For each definition it:

- fits class-balanced recording-level logistic probes to every WavLM layer,
  using temporal means alone and means plus temporal standard deviations;
- selects the layer, pooling mode, and regularization using subject-grouped
  training cross-validation only;
- compares globally pooled recording features against a window-level classifier;
- gives every recording equal total weight in window-level training;
- selects mean or top-quartile window-probability aggregation using training OOF
  predictions only; and
- reports external-validation AUROC, average precision, balanced accuracy,
  macro F1, MCC, sensitivity, and specificity with subject-clustered bootstrap
  confidence intervals.

The existing validation set is never used for model, layer, regularization,
aggregation, or probability-cutoff selection.

Run a configuration check:

```bash
./scripts/run_wavlm_binary_audit.sh --dry-run
```

Run the experiment:

```bash
./scripts/run_wavlm_binary_audit.sh
```

This experiment uses cached embeddings and scikit-learn, so it does not require
the GPU. Results are written to `outputs/wavlm-base-plus-binary-audit/`, with the
main summary in `report.md`.
