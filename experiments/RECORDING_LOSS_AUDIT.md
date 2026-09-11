# Recording loss audit

Rank the recordings evaluated by a checkpoint from highest to lowest loss:

```bash
uv run python scripts/rank_recording_losses.py \
  outputs/hpo_text_regression/confirmation/trial-0004-seed-43/best_checkpoint.pt
```

The checkpoint validation manifest and preprocessing configuration are used by
default. The complete ranked table is saved as `recording_loss_audit.csv` next
to the checkpoint. It includes the audio, word-timestamp, and diarization paths,
labels, predictions, window count, and each loss component.

Audit a different manifest or ranking component:

```bash
uv run python scripts/rank_recording_losses.py path/to/best_checkpoint.pt \
  --manifest path/to/manifest.csv \
  --rank-by absolute-error \
  --top-k 40 \
  --output outputs/my_recording_audit.csv
```

Available ranking choices are `combined`, `regression`, `absolute-error`, and
`classification`. Regression loss is squared error on standardized targets,
matching training. Combined loss applies the checkpoint task and class weights.

Inspect the resolved configuration without loading the model or running
inference:

```bash
uv run python scripts/rank_recording_losses.py path/to/best_checkpoint.pt --dry-run
```
