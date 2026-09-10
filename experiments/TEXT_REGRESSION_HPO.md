# MentalBERT-only regression HPO

This resumable Optuna study trains only the regression objective with
recording-level sampling. It searches embedding normalization, whether text
LoRA is enabled, conditional LoRA rank and alpha, gradient accumulation,
learning rate, weight decay, window length, and validation stride. The LR
scheduler is disabled.

All window-length trials retain the cohort eligible for 30-second windows, so
their validation R-squared values cover the same recordings. The known
four-window configuration is queued as trial 0. LoRA rank and alpha are only
sampled when LoRA is enabled because they otherwise have no effect.

Validate the setup without launching training:

```bash
./scripts/run_text_regression_hpo.sh --dry-run
```

Run the default overnight search:

```bash
./scripts/run_text_regression_hpo.sh
```

The defaults target 20 trials and stop starting new trials after 10 hours. An
active trial is allowed to finish. Both limits can be changed:

```bash
./scripts/run_text_regression_hpo.sh --n-trials 30 --timeout-hours 12
```

Repeating the same command resumes the SQLite study. Results are written to
`experiments/hpo_text_regression/report.md`, with individual run artifacts
under `outputs/hpo_text_regression/screening/`.
