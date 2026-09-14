# openSMILE IS09 acoustic baselines

This experiment reproduces the feature configuration found in the colleague
code at `rdino_model/Ross_codes_Sept01/codes/extract_opensmile_features_mod.py`:
`config/is09-13/IS09_emotion.conf`. It uses the equivalent official Python
openSMILE `FeatureSet.IS09` implementation, which emits 384 utterance-level
functionals.

IS09 features are extracted from the same diarization-aware 10-second windows
used for the WavLM audits, with preserved gaps and a 7.5-second stride. The
window features are aggregated to each recording using either:

- the mean of every IS09 functional; or
- the mean and between-window standard deviation.

The following recording-level models are evaluated:

- Elastic Net regression for continuous MADRS;
- random-forest regression for continuous MADRS;
- elastic-net logistic regression for MADRS ≥20; and
- random-forest classification for MADRS ≥20.

Pooling and model hyperparameters are chosen with subject-grouped training CV.
The external validation set is never used for selection. AUROC and R² confidence
intervals use subject-clustered bootstrap resampling.

Inspect the commands:

```bash
./scripts/run_opensmile_is09_baselines.sh --dry-run
```

Run the standalone CPU experiment:

```bash
./scripts/run_opensmile_is09_baselines.sh
```

Useful environment overrides are `AUDIO_WORKERS`, `SMILE_WORKERS`, `JOBS`, and
`OUTPUT_DIR`. The main result is written to
`outputs/opensmile-is09-baselines/report.md`. Fitted estimators, validation
predictions, selected settings, and the 100 largest feature importances from
each model are also retained.

## Paper-aligned 60-second variant

The follow-up experiment extracts IS09 once from a single 60-second crop per
recording. It selects the earliest primary-speaker run long enough for the crop,
preserves unlabeled gaps within that run, and never crosses a detected speaker
switch. This yields exactly 384 recording-level features rather than pooling
functionals computed on shorter windows.

Inspect or run it with:

```bash
./scripts/run_opensmile_is09_paper_aligned.sh --dry-run
./scripts/run_opensmile_is09_paper_aligned.sh
```

It writes to `outputs/opensmile-is09-paper-aligned` by default and does not
overwrite the original windowed experiment. On the current manifests, 453 of
461 training recordings and 117 of 118 validation recordings provide an
eligible continuous crop.
