# Shared evaluation protocol

The generated bundle for the September 13 source CSV is
`outputs/evaluation/20260913_v3/`. It contains 891 recordings from 515 subject IDs
across CDRIN, FORBOW, OPTD, TIDE, and VMP. The source CSV remains unchanged.
All recordings are retained. Classification labels are copied but are not used
to select speakers, define folds, or calculate metrics.

This is a model-independent data and evaluation contract. The existing
`train.py` and older extraction scripts **do not automatically consume it**:
they construct their own windows, can drop short runs, and use their previous
aggregation rules. Use the bundle's windows and folds when integrating a model;
passing only a new recording CSV to the old dataset is insufficient. No training
has been launched or existing experiment configuration changed. The new
`scripts/run_mentalbert_loso.sh` launcher consumes this bundle directly; see
[MENTALBERT_LOSO.md](MENTALBERT_LOSO.md).

## Speaker and timing rules

1. Clip diarization to the actual audio duration. Select the speaker with the
   greatest unioned speaking duration, before filling gaps. Segment count is not
   a reliable measure of the dominant speaker. Ties are resolved by speaker label
   and flagged for review.
2. Preserve the original audio between consecutive primary-speaker intervals
   when no other speaker occupies the gap and it is at most 60 seconds long.
   Gaps strictly longer than 60 seconds are cropped out, leaving independent
   blocks on either side; they are never joined after cropping. Leading/trailing unlabeled time and
   unlabeled time at speaker changes are not assigned to either speaker.
3. Every non-primary speaker interval is excluded and breaks a block. If speakers
   overlap, exclude the overlap too, including other speech nested within a
   longer primary-speaker segment. Speaker labels other than the primary label
   are all treated as other speakers.
4. Store each block as an original-audio start/end pair. Never concatenate the
   crops before feature extraction. Compute pause statistics separately within
   each block; do not treat the separation between blocks as a subject pause.
   The preserved unlabeled gaps are candidate pauses or diarizer-missed speech,
   not verified silence. Use an appropriate speech/voicing detector for actual
   pause measurement.
5. Retain words fully contained in a subject block. Omit words crossing a speaker
   boundary conservatively; keep original timestamps. This does not guarantee
   perfect speaker attribution when diarization/ASR itself is wrong.

For example, subject `[1,3]`, subject `[5,9]`, interviewer `[10,11]`, subject
`[12,20]` becomes subject blocks `[1,9]` and `[12,20]`. The pause from 3 to 5
seconds remains intact. The two blocks are processed independently.

The primary-speaker heuristic needs review for single-speaker files, nearly tied
speakers, and recording issues. `review_queue.csv` contains these
flags along with timestamp clipping and recordings without a 55-second block.
Flags do not exclude recordings. Gaps from 30 through 60 seconds are preserved
and are not flagged for their length. Per the collection protocol and the
confirmed microphone dropout, gaps over 60 seconds are removed and logged as
`long_gap_cropped`. The v3 bundle removes one 250.914-second gap; other subject
audio from that recording remains included. After reviewing the original audio, provide a JSON
mapping of `recording_id` to corrected speaker label via `--speaker-overrides`
and build a new version. This corrects speaker choice, not a diarizer that merged
two real people into one label; that requires corrected diarization input.

## Windows and aggregation

`blocks.jsonl` is the canonical timeline with subject words. `windows.csv` is a
ready-to-use default view: at most 55 seconds, without overlap, with a boundary
moved backward when necessary to avoid splitting a word. Short blocks and short
tails are retained. A window never spans two subject blocks.

The model processes each window independently. Regression predictions are
duration-weighted within each recording. The provided weights sum to one per
recording, so a very brief backchannel does not receive the same weight as a long
speech block. Evaluate once per recording, with sensitivity to unequal visit
counts shown by equal-subject MAE. Train with a recording-level objective or
explicitly recording-balanced sampling; do not let window count silently set a
recording's training weight.

Two predefined window views keep the SAME recording cohort:

- Audio/all: every window, including windows with no words.
- Text: `word_count > 0`; normalize duration weights over these windows within
  each recording. The scorer implements this with `--window-view text`.

The all view contains 16,636 windows; 2,381 have no retained words. Those empty
windows collectively contain about 520 seconds, approximately 0.10% of retained
audio duration. Every recording still has nonempty subject text.

`clinical_evaluation.load_window_audio(window)` reads a contiguous crop and
returns a mono, 16-kHz tensor plus its true sample length. Resampling happens
within the crop, never across interviewer boundaries. Do not repeat short
waveforms. Model-specific batching must handle padding, masks, and any minimum
frontend length explicitly; very short blocks can be under 20 ms. Undefined
voice-quality measurements need a predefined missing-feature policy. Padding
must not enter pause statistics. For text, inspect tokenizer lengths and split
within blocks if needed rather than silently truncating valuable content.

These windows/weights differ from the older overlapping-window protocol, so
the previous R² of 0.32 is not a directly comparable reference. Recompute all
candidate models on the same new recording cohort and folds. Alternative window
sizes or token-based subdivisions can be separate declared configurations, but
must preserve the subject blocks and recording cohort.

## Folds and model selection

`loso/<study>/test.csv` holds out that study. `train.csv` contains the other
studies, with any subjects shared with the test study purged. There are no
cross-study subject IDs in this CSV; the implementation still handles them.

Each outer fold has four `inner/<validation-study>/` train/validation pairs.
Select hyperparameters and stopping schedules using only these inner folds,
then fit on the outer training recordings and predict the outer test study once.
For neural networks, choose the final training schedule using inner validation,
not outer-test early stopping. Fit scaling, PCA, feature selection, imputation,
target normalization, and any learned fusion only within the appropriate
training partition. Late fusion must train on out-of-fold predictions.

`mixed_subject/fold_0/` through `fold_4/` provide a secondary in-mixture benchmark
with inner subject folds. Assignment is deterministic, balances subject counts
within study, and ignores MADRS and class labels. Keep these results separate
from the primary LOSO results. IDs are taken literally from the CSV: grouping
does not establish whether two different IDs actually identify the same person.

Record study-wise MAE, RMSE, R², Pearson correlation, mean error, and calibration
slope/intercept. Equal-study MAE/RMSE are primary summaries; pooled metrics and
equal-subject MAE are supplementary. The baseline is the outer TRAINING mean,
not the observed mean of the held-out study. Calibration fits on test predictions
are descriptive diagnostics only, never a deployable recalibration result.

The current scorer produces point estimates. For later model comparisons, use
paired subject-cluster resampling, preserving visits and study membership; with
only five studies, those intervals do not quantify uncertainty over all possible
future studies. Repeatedly inspecting these outer folds turns them into
development data. A new untouched cohort is the final transportability test.

## Build and score

Run from the repository root using its existing environment:

```bash
.venv/bin/python scripts/build_evaluation_bundle.py \
  --source /data/Clinical_vars/rdino_bert_full_20260913.csv \
  --output outputs/evaluation/20260913_v4 \
  --max-gap-seconds 60
```

The builder refuses to overwrite a directory, errors on invalid records instead
of silently dropping them, and records source/sidecar/code hashes and output
checksums. Audio files are identified by size/mtime plus their resolved paths;
their full contents are not hashed. Generated files contain local clinical text
and live under the repository's ignored `outputs/` directory.

For a completed five-fold LOSO run, combine test predictions into one CSV:

```csv
recording_id,test_study,prediction
<recording_id>,CDRIN,18.2
```

Use exactly one row per recording, in original MADRS units. For window-level
predictions, use `window_id,test_study,prediction` and `--level window` instead.
Text-only windows additionally use `--window-view text`.

```bash
.venv/bin/python scripts/score_evaluation_bundle.py \
  --bundle outputs/evaluation/20260913_v3 \
  --predictions path/to/loso_predictions.csv \
  --output outputs/evaluation_scores/my_model
```

The scorer verifies the bundle checksums and rejects duplicate, missing,
nonfinite, or wrong-study predictions. It writes `metrics.json` and
`recording_predictions.csv`, retaining IDs for paired comparisons. It checks
declared test-fold membership; it cannot prove that an external training process
did not use test data.

Validation:

```bash
.venv/bin/python -m unittest discover -s tests -v
```
