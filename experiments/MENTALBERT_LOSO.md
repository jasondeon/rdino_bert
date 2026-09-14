# MentalBERT on the shared evaluation bundle

This is a single-configuration, regression-only experiment on
`outputs/evaluation/20260913_v3/`: all 891 recordings, 515 subject IDs, five
studies. Gaps up to 60 seconds remain intact; longer gaps are cropped out and
split the subject block. The one removed microphone dropout is 250.914 seconds.
The 30-second review trigger is removed. Existing study/subject assignments and
source files are unchanged.

The new runner consumes the bundle directly. It does not invoke `train.py`, load
RDINO, read audio during training, or re-derive eligibility or window boundaries.
Classification labels are unused. No clinical-data training has been launched.

## Run

From the repository root:

```bash
bash scripts/run_mentalbert_loso.sh --dry-run
bash scripts/run_mentalbert_loso.sh
```

The first command verifies bundle checksums and subject separation, confirms
that the model files are cached, and tokenizes every subject-text window on the
CPU. It writes a preflight report without loading model weights or training.
The second command runs on CUDA and writes to
`outputs/mentalbert-loso-20260913/`. It needs no active Codex session.

For a detached run with a persistent console log, after the preflight:

```bash
nohup bash scripts/run_mentalbert_loso.sh \
  > outputs/mentalbert-loso-20260913/training.log 2>&1 &
```

All fits run sequentially on one device. By default this is **20 inner fits plus
five final fits**, not five fits alone. Each inner fit runs the same 150-epoch
budget, validating every five epochs. This avoids comparing unequal or
incompletely observed stopping schedules. There is no hyperparameter grid.

To run one complete outer fold first:

```bash
bash scripts/run_mentalbert_loso.sh --study CDRIN
```

Rerun without `--study` to finish the rest. Fold selection is operational and
does not change the experiment configuration. Completed folds are verified and
skipped; interrupted fits resume from their last completed epoch. An interrupted
epoch is repeated. A process lock prevents concurrent writers to the same run.

Changing configuration, bundle contents, code, or model revision requires a new
output directory, even after preflight. For example, an explicitly shorter
development run can use:

```bash
bash scripts/run_mentalbert_loso.sh \
  --max-epochs 50 --output outputs/mentalbert-loso-50epochs
```

Do not choose between such runs using the outer test scores while describing
those same scores as an untouched final estimate. If the selected epoch is at
the maximum, `epoch_selection.json` flags that the search budget may be limiting.

## Model and objective

Defaults use the saved trial-36 configuration that achieved validation R²
0.325163 in `experiments/hpo_text_regression/report.md`:

| Setting | Value |
| --- | --- |
| Backbone | `mental/mental-bert-base-uncased`, revision pinned at preflight |
| LoRA | Rank 8, alpha 8, both feed-forward dense layers per BERT block |
| Text pooling | Attention-mask-aware mean pooling |
| Normalization | BatchNorm, as in trial 36 |
| Head | Hidden size → 200 → 50 → 1, SiLU, dropout 0.1 |
| Optimizer | AdamW, LR 1.1962402885307259e-5, weight decay 1e-6 |
| Schedule | Constant LR; inner validation selects epoch count |
| Training batch | Two recordings × four sampled windows |
| Regression target | Standardized using the current training recordings only |
| Loss | MSE on the mean sampled-window prediction for each recording |
| Precision | float32 |
| Seed | 40, with deterministic stage-specific seeds |

This is not an exact reproduction of trial 36: the canonical windows are now
at most 55 seconds and nonoverlapping, short blocks are retained, the training
cohort is larger, and sampling/aggregation use duration weights. The historical
settings also come from prior exploration involving these data; the new LOSO
analysis is a development/transportability audit, not a pristine prospectively
untouched estimate.

Every training recording appears once per epoch. Four windows are drawn with
replacement in proportion to their duration among that recording's nonempty
text windows. Their predictions are averaged before computing one loss. This
is a stochastic approximation to full-recording MSE; finite sampling also adds
prediction-variance noise to the objective. It keeps training memory bounded
and avoids making long or fragmented recordings dominate the training set.

Validation/test use every nonempty-text window, with exact duration-weighted
aggregation. Text is never concatenated across subject blocks. If a window
exceeds BERT's token capacity, it is subdivided into nonoverlapping token chunks
inside that window and all tokens are retained. Chunk predictions are weighted
by content-token count before recovering the canonical window prediction.

The upstream checkpoint can warn that its unused BERT pooler weights are
newly initialized. This model uses attention-masked mean pooling of token
states instead; that pooler does not contribute to predictions.

BatchNorm uses training-batch statistics during training and learned running
statistics during validation/test. Its buffers are included in checkpoints.
The runner requires at least two sampled windows when using BatchNorm, even
for a final minibatch containing only one recording. `--normalization layernorm`
is available as a separate declared experiment, not an automatic change.

## Selection and held-out evaluation

For each outer test study:

1. Train four fresh models, each withholding a different inner study. Fit
   target standardization separately on each inner training set.
2. At the common evaluated epochs, average inner-study RMSE with equal weight
   per study. Select the minimum; break ties in favor of the earlier epoch.
3. Initialize a fresh model and refit on all four outer training studies for
   exactly that epoch count, with freshly fitted training-target scaling.
4. Predict the outer study once, restoring original MADRS units. No test-based
   clipping, recalibration, stopping, or epoch selection is applied.

The combined scorer checks every expected text window and all 891 recordings,
then writes study-wise metrics, equal-study MAE/RMSE, pooled metrics,
equal-subject MAE, and the outer-training-mean baseline.

## Artifacts and recovery

- `experiment_config.json`: exact settings, bundle/code checksums, model revision.
- `preflight.json`: data/tokenizer audit; its `training_started: false` describes
  the preflight itself, not the later lifecycle of the experiment.
- `<study>/inner/<validation-study>/history.json`: validation learning curve.
- `<study>/epoch_selection.json`: mean inner-study RMSE curve and selected epoch.
- `<study>/final/last_checkpoint.pt`: final adapters, MLP/normalization parameters
  and buffers, training-target scaler, optimizer, RNG state, and configuration.
- `<study>/window_predictions.csv` and `recording_predictions.csv`: held-out outputs.
- `loso_window_predictions.csv`, `loso_recording_predictions.csv`, `metrics.json`:
  combined results once all five outer folds finish.

Checkpoints contain only trainable model state and required buffers, plus the
optimizer/RNG/scaler metadata. Frozen backbone weights are restored from the
pinned model revision. MentalBERT uses separate pinned commits for its original
config/tokenizer and the cached `refs/pr/5` safetensors conversion; both appear
in `experiment_config.json`. The model cache is required on resume/inference;
`--allow-download` is an explicit alternative for a fresh environment.

Validation includes synthetic boundary tests, token overflow, duration sampling,
gradient propagation through aggregation, BatchNorm buffer restoration, and a
complete nested experiment with a tiny randomly initialized BERT on the CPU.
A separate check also loads the actual cached MentalBERT and verifies finite
CPU forward/backward results on two synthetic sentences, with no optimizer
step. These tests do not measure clinical accuracy or GPU throughput.

```bash
.venv/bin/python -m unittest discover -s tests -v
```
