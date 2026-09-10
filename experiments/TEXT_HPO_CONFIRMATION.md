# MentalBERT HPO confirmation

This suite compares HPO trials 36 and 4 on three new paired seeds: 41, 42,
and 43. Both configurations remain MentalBERT-only, regression-only, use
recording-level loss with four sampled windows, and evaluate the same cohort
eligible for 30-second windows.

Inspect all six commands without launching training:

```bash
./scripts/run_text_hpo_confirmation.sh --dry-run
```

Run the suite independently of Codex:

```bash
./scripts/run_text_hpo_confirmation.sh
```

Completed runs carry a `.complete` marker and are skipped when the launcher is
repeated. An incomplete run is never overwritten automatically; move its
directory aside before resuming.

Outputs are stored under `outputs/hpo_text_regression/confirmation/`. After all
runs, the launcher writes `report.md`, `confirmation_runs.csv`, and
`confirmation_summary.csv` there. To regenerate the report without training:

```bash
uv run python scripts/summarize_text_hpo_confirmation.py \
  --output-root outputs/hpo_text_regression/confirmation \
  --screening-root outputs/hpo_text_regression/screening
```
