# RDINO frozen-embedding window-size audit

## Question

Does the amount of audio presented to frozen RDINO change the amount or geometry
of recording-level regression signal in its embeddings?

The local RDINO training recipe used approximately four-second global crops (and
shorter local crops), so the existing 20-second input may be substantially outside
the checkpoint's pretraining regime. This audit compares 5, 10, 20, and 30 seconds.

## Controls

- The train and validation manifests remain subject-disjoint and unchanged.
- All conditions require 30 seconds of eligible primary-speaker audio, ensuring
  that every condition evaluates the same recording cohort.
- Unlabeled gaps are preserved, while detected non-primary-speaker intervals remain
  hard boundaries.
- RDINO is frozen. The only fitted model is a ridge probe whose regularization is
  selected by subject-grouped cross-validation on the training split.
- Validation uses all deterministic windows, with stride set to 75% of window size.
- The completed corrected 20-second audit is reused as the reference rather than
  recomputed.

## Run

The corrected 20-second audit must finish first:

```bash
./scripts/run_rdino_embedding_audit.sh
```

Inspect the commands without using the GPU:

```bash
./scripts/run_rdino_window_size_audit.sh --dry-run
```

Then run the sequential single-GPU suite independently of Codex:

```bash
chmod +x scripts/run_rdino_window_size_audit.sh
./scripts/run_rdino_window_size_audit.sh
```

Results are written to `outputs/rdino-window-size-audit/`, including:

- `report.md`: compact results and interpretation;
- `window_size_summary.csv`: probe metrics and paired R² differences from 20s;
- `window_size_geometry.csv`: within/between-recording embedding geometry;
- `window_size_audit.png`: absolute R² and paired differences from 20s.

## Classification-label UMAP

Generate comparable recording-level UMAPs for the train and validation splits:

```bash
uv run python scripts/plot_rdino_embedding_umap.py
```

The script averages windows within each recording and fits an independent cosine
UMAP for each duration. Axes and orientation therefore cannot be compared across
panels; each panel is intended only to show its own class organization. Class labels
are used only for color, never during fitting. The script also writes silhouette
scores in the original 512-dimensional space so that projection artifacts are not
mistaken for class separation.

An interrupted run never overwrites partial embeddings. Completed window sizes are
skipped when the suite is restarted; move an incomplete per-window directory aside
before retrying it.
