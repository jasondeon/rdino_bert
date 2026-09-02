# BERT + RDINO training

This folder is a clean replacement for the experimental scripts in
`../Ross_codes_Sept01/codes`. It does not modify or depend on their dataset
filename conventions.

## Inputs

Create separate training and validation CSV manifests with these columns:

```csv
audio_path,word_timestamps_path,class_label,regression_label,subject_id
audio/001.wav,transcripts/001.words.json,1,-0.32,001
audio/002.wav,transcripts/002.words.json,0,0.08,002
```

- `audio_path` may be absolute or relative to the CSV file.
- `word_timestamps_path` points to JSON, CSV, or TSV word timestamps. Paths may
  be absolute or relative to the manifest.
- `class_label` must be an integer in `[0, num_classes)`.
- `regression_label` is the value used directly by MSE loss. If the source
  column is percentage change, divide it by 100 while building the manifest.
- `subject_id` is optional, but recommended for auditing subject-independent
  train/validation splits.

Each recording is expanded into fixed-length, overlapping windows. By default,
windows are 55 seconds long and advance by 10 seconds; the final window is
anchored at the end so the recording tail is retained. Each window's transcript
contains the words whose timestamp intervals overlap that window. Windows with
no words are omitted unless `--include-empty-text` is passed. Labels remain at
recording level and are copied onto every generated window.

Audio is converted to mono and resampled to 16 kHz. Recordings shorter than the
configured window are excluded from both training and validation; a recording
exactly as long as the window produces one example. The model uses 3D-Speaker's own
80-dimensional Kaldi-style `FBank` processor with mean normalization, matching
the pretrained RDINO inference pipeline.

Supported word timestamp JSON forms include a direct list, `{"words": [...]}`
and Whisper-style `{"segments": [{"words": [...]}]}`. A word entry may use:

```json
{"word": "hello", "start": 12.31, "end": 12.72}
```

CSV/TSV accepts `word`, `text`, or `token`, together with `start`/`end` (or
`start_time`/`end_time`) columns. Times are seconds from the start of the audio.

## RDINO prerequisites

The 3D-Speaker source is tracked as the `vendor/3D-Speaker` Git submodule. Since
upstream does not provide Python package metadata, register its source in the uv
environment after cloning or recreating the environment:

```powershell
git submodule update --init --recursive
uv sync
uv run python scripts/install_speakerlab.py
```

The model also needs:

1. `assets/rdino.yaml` (included here from the supplied file), and
2. `assets/pretrained_rdino.pth`, containing a `teacher` state dictionary.

The checkpoint is not currently present and must be obtained separately.

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
