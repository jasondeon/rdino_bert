# BERT + RDINO training

## Inputs

Create separate training and validation CSV manifests with these columns:

```csv
audio_path,word_timestamps_path,diarization_path,class_label,regression_label,subject_id
audio/001.wav,transcripts/001.words.pkl,diarization/001.pkl,1,-0.32,001
audio/002.wav,transcripts/002.words.pkl,diarization/002.pkl,0,0.08,002
```

- `audio_path` may be absolute or relative to the CSV file.
- `word_timestamps_path` points to JSON, CSV, or TSV word timestamps. Paths may
  be absolute or relative to the manifest.
- `diarization_path` points to pickle, JSON, CSV, or TSV speaker segments. Paths
  may be absolute or relative to the manifest.
- `class_label` must be an integer in `[0, num_classes)`.
- `regression_label` is the value used directly by MSE loss. If the source
  column is percentage change, divide it by 100 while building the manifest.
- `subject_id` is optional, but recommended for auditing subject-independent
  train/validation splits.

The speaker with the greatest total diarized duration is selected independently
for each recording. Consecutive diarization segments belonging to that speaker
are concatenated into a run, and fixed-length, overlapping windows are created
within each run. A segment belonging to another speaker always breaks the run,
so a window never crosses another speaker's turn. By default, windows are 55
seconds long and advance by 10 seconds; the final window within each eligible
run is anchored at the run's end. Runs shorter than the window length are
discarded. Windows with no words are omitted unless
`--include-empty-text` is passed. Labels remain at recording level and are copied
onto every generated window.

Audio is converted to mono and resampled to 16 kHz. Recordings shorter than the
configured window are excluded from both training and validation; a recording
exactly as long as the window produces one example. The model uses the 80-dimensional mel-spectrogram frontend from the supplied
RDINO training and inference recipe.

Word timestamp pickle files may contain tuples such as
`[("We", 25.08, 25.16), ("are", 25.16, 25.24)]`. Supported JSON forms include a direct list, `{"words": [...]}`
and Whisper-style `{"segments": [{"words": [...]}]}`. A word entry may use:

```json
{"word": "hello", "start": 12.31, "end": 12.72}
```

CSV/TSV accepts `word`, `text`, or `token`, together with `start`/`end` (or
`start_time`/`end_time`) columns. Times are seconds from the start of the audio.

Diarization pickle files may contain tuples such as
`[(24.97, 29.73, "SPEAKER_00"), (30.05, 33.00, "SPEAKER_01")]`.
JSON objects and CSV/TSV rows may use `speaker`, `speaker_id`, or `label` with
the same timestamp column names supported for word timestamps. Speaker labels
are treated as arbitrary identifiers.

## RDINO prerequisites

The 3D-Speaker source is tracked as the `vendor/3D-Speaker` Git submodule. Since
upstream does not provide Python package metadata, register its source in the uv
environment after cloning or recreating the environment:

```powershell
git submodule update --init --recursive
uv sync
uv run python scripts/install_speakerlab.py
```

Place the ModelScope `pretrained_rdino.pth` checkpoint in `assets/`. The model
defaults to the gated `mental/mental-bert-base-uncased` checkpoint; accept its
Hugging Face access conditions and authenticate locally before the first run.
You can temporarily pass `--text-model bert-base-uncased` to use ordinary BERT.

The trainable architecture follows the supplied design: rank-2 LoRA on both
MentalBERT feed-forward layers, rank-2 LoRA on RDINO's terminal attentive-pooling
and embedding transformations, separate batch normalization of the 768-D text
and 512-D audio embeddings, shared `1280 -> 200 -> 50` feed-forward layers, and
four-class plus regression heads.

## Environment

The project uses Python 3.11 and uv. Recreate or update the environment with:

```powershell
uv sync
uv run python scripts/install_speakerlab.py
```

The upstream requirements pin old NumPy and scikit-learn versions that conflict
with the current Python 3.11 environment, so they are not installed wholesale.
The SpeakerLab components used here import successfully with this project's
dependency set.

## Validate the manifests

This checks columns, labels, paths, and split overlap without loading models:

```powershell
python validate_manifest.py --train train.csv --validation validation.csv
```

## Train

```powershell
python train.py `
  --train-manifest C:\path\train.csv `
  --validation-manifest C:\path\validation.csv `
  --rdino-checkpoint C:\path\pretrained_rdino.pth `
  --window-seconds 55 `
  --stride-seconds 10
```

Use `python train.py --help` for all options. Checkpoints include the model
configuration and text-model identifier, rather than only parameter tensors.

Pass `--augment-audio` to apply runtime audio augmentation to training windows
only. By default, AWGN is applied with probability 0.5 at a uniformly sampled
10--30 dB SNR, and synthetic room reverb is applied independently with
probability 0.3. Reverb uses a randomized 0.2--0.8 second RT60 and 0.1--0.4 wet
mix. All ranges and probabilities have corresponding command-line options.
Validation and inference waveforms remain clean. The settings are written to
`audio_augmentation.json` and saved in the best checkpoint.

Pass `--stride-jitter-seconds 2` to randomly shift each training window start by
up to two seconds on every access. The shifted audio and word-timestamp transcript
remain aligned, and the crop is clamped to its original uninterrupted
primary-speaker run. Validation windows are never jittered. The nominal window
count and stride remain unchanged.

Pass `--standardize-regression-labels` to fit the regression-label mean and
population standard deviation on the training recordings only. Validation
targets are kept on their original scale, and predictions are converted back to
that scale for the prediction CSV. `regression_rmse` is computed in standardized
units, while `regression_rmse_original_scale` is also reported. Both standardized
and original-scale values are written to the prediction CSV. The fitted values
are stored in `regression_standardization.json` in the output directory and in
each checkpoint under `regression_standardization`. At inference, restore the
model's raw regression output with `prediction * std + mean`.


Validation metrics are aggregated at the recording level: class probabilities
and regression outputs are averaged across a recording's windows before scoring.
Window predictions are written to `predictions_epoch_N.csv`, while aggregated
predictions are written to `recording_predictions_epoch_N.csv`.

Frozen RDINO BatchNorm layers keep their pretrained running mean and variance by
default, even while LoRA adapters are training. Pass
`--update-rdino-batchnorm-stats` only to reproduce the earlier behavior in which
those buffers changed during downstream training. The separate modality embedding
normalizers remain trainable. Select them with
`--embedding-normalization batchnorm` (the original architecture) or
`--embedding-normalization layernorm` (batch-independent).

Use `--modality text`, `--modality audio`, or `--modality both` for modality
ablations. `--disable-text-lora` and `--disable-rdino-lora` keep the corresponding
pretrained backbone completely frozen. LoRA rank and alpha are exposed through
`--lora-rank` and `--lora-alpha`.

Pass `--log-task-gradients` to measure classification and regression gradient
norms and cosine similarity on the shared trainable parameters for the first
batch of every epoch. Results are written to `gradient_diagnostics.csv`, both for
all shared parameters and separately for fusion, normalization, and active LoRA
adapter groups. Increase the number of sampled batches with
`--task-gradient-batches-per-epoch`; the diagnostic suite uses ten. Negative
cosine similarity indicates conflicting task directions; the weighted gradient-
norm ratio shows which task dominates after applying the
configured loss weights.

Training stops by default after five epochs without improvement in
recording-level regression R² and writes the best model to `best_checkpoint.pt`.
Adjust this with `--early-stopping-patience` and `--early-stopping-min-delta`.

A plateau scheduler monitors recording-level validation R² by default. After its
configured patience it multiplies the learning rate by 0.5, down to `1e-7`.
Configure it with `--lr-scheduler-factor`, `--lr-scheduler-patience`, and
`--min-learning-rate`, or disable it with `--lr-scheduler none`. The per-epoch
learning rate is stored in `training_history.csv`; scheduler configuration and
state are saved in `lr_scheduler.json` and `best_checkpoint.pt`.

Classification losses use square-root inverse-frequency class weights by default.
The frequencies are fitted only on eligible training examples and match the active
sampling unit: recordings with `--train-sampling recording` or
`--train-sampling balanced_window`, and windows with `--train-sampling window`.
The weights are normalized to have mean sample weight
one and saved to `classification_weighting.json` and the best checkpoint. Pass
`--class-weighting none` to recover unweighted classification loss.

The supplied RDINO recipe uses the mel-spectrogram frontend.

Each epoch is appended to `training_history.csv` in the output directory. Plot
the training/validation losses and validation metrics after or during a run with:

```bash
uv run python plot_training.py --output-dir outputs/main-run
```

This writes `outputs/main-run/training_curves.png`.

Training uses recording-level multi-window loss by default. Four eligible windows
are sampled per recording, class probabilities and regression outputs are averaged,
and one classification plus regression loss is computed per recording. Recording
order and window selections are shuffled each epoch; recordings with fewer than
four windows are sampled with replacement rather than discarded. Set the group
size with `--train-windows-per-recording`. `--batch-size` counts windows and must
be divisible by the group size (for example, batch size 8 with four windows gives
two recordings per optimizer batch). Pass `--train-sampling window` to restore
shuffled window-level training.

`--train-sampling balanced_window` is a third mode. It retains ordinary
window-level losses and the same total number of samples per epoch as `window`,
but gives every eligible recording the same number of sampled windows (within one
window when the total is not divisible). Selected windows are globally shuffled,
so batches do not intentionally group windows from the same recording. This is a
cleaner test of recording-balanced sampling than the grouped recording-level loss.

## Structural diagnostic suite

Run the focused single-GPU suite with:

```bash
./scripts/run_diagnostic_experiments.sh
```

It sequentially runs six single-seed experiments: a frozen-RDINO-BatchNorm
reference, the balanced-window/LayerNorm multimodal model, text-only, audio-only,
no LoRA, and ordinary BERT instead of MentalBERT. It never launches concurrent
GPU jobs. Completed output directories are skipped on a later invocation, while
an incomplete directory must be moved aside explicitly before retrying.

Outputs are placed under `outputs/diagnostic-suite` by default. Override that
without editing the script with, for example:

```bash
OUTPUT_ROOT=outputs/diagnostic-suite-2 ./scripts/run_diagnostic_experiments.sh
```

After the final run, `report.md`, `summary.csv`, and
`gradient_summary_by_group.csv` summarize model performance, prediction shrinkage,
rare-class recall, disagreement between heads, and task-gradient interaction.
