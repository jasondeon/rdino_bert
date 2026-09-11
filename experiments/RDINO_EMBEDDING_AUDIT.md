# Frozen RDINO embedding audit

This audit determines whether the pretrained 512-dimensional RDINO embeddings
contain usable recording-level regression signal before making further neural-model
changes.

It performs two stages:

1. Extract and cache frozen RDINO embeddings for every eligible training and
   validation window on the GPU.
2. On the CPU, aggregate them by recording and fit ridge probes using either the
   across-window mean or concatenated mean and standard deviation.

Ridge regularization is selected using subject-grouped cross-validation on training
recordings only. The analysis refuses to run if a subject appears in both training
and validation caches. It reports R2, RMSE, ICC(2,1), Pearson correlation,
prediction variance, paired bootstrap intervals, and within-versus-between-recording
embedding similarity.

From the repository root, inspect the commands without doing work:

```bash
./scripts/run_rdino_embedding_audit.sh --dry-run
```

Run the audit without Codex attached:

```bash
./scripts/run_rdino_embedding_audit.sh
```

Results are written to `outputs/rdino-embedding-audit-preserved-gaps/report.md`,
with cached arrays, CSV tables, predictions, and a scatter plot in the same
directory. This new name prevents reuse of embeddings cached under the legacy
gap-concatenation policy. The extraction is the expensive stage; once cached,
`scripts/analyze_rdino_embeddings.py` can be rerun cheaply with different probe
settings.
