# RDINO structural experiment suite

This suite asks five targeted questions before doing another hyperparameter search:

1. Was the earlier audio-only failure mainly caused by its approximately `3e-6`
   learning rate and window-level multitask setup?
2. Does the shallower `512 -> 50` head from the colleague's audio model generalize
   better than the current `512 -> 200 -> 50` head?
3. Does a faster learning rate for the randomly initialized normalization/head help
   while retaining a conservative RDINO-adapter rate?
4. Does LoRA need to adapt all shape-safe 1x1 channel-mixing convolutions throughout
   ECAPA-TDNN rather than only the terminal pooling/output convolutions? Dilated
   temporal filters stay frozen because PEFT's Conv1d adapter does not preserve
   their output geometry.
5. Are frozen pretrained RDINO embeddings already useful when paired with the
   colleague-style fast shallow head?

All arms use audio only, recording-level regression loss, the same eligible cohort,
the same seed, and all validation windows. Classification, augmentation, stride
jitter, and the LR scheduler are disabled so they cannot obscure these comparisons.
RDINO BatchNorm running statistics remain frozen.

Run from the repository root:

```bash
./scripts/run_rdino_structural_experiments.sh
```

Inspect commands without launching training:

```bash
./scripts/run_rdino_structural_experiments.sh --dry-run
```

To store results elsewhere:

```bash
OUTPUT_ROOT=outputs/rdino-structural-2 ./scripts/run_rdino_structural_experiments.sh
```

The runner executes sequentially on one GPU, refuses to overwrite interrupted runs,
plots each completed run, and writes `summary.csv` plus `report.md` after all five
runs finish. If a run is interrupted, move its incomplete output directory aside
before restarting the suite.

## Final segment-level objective test

After the frozen-embedding audit, run the last regression-only comparison with:

```bash
./scripts/run_rdino_balanced_window_final.sh
```

This writes run `06-balanced-window-preserved-gaps`. It retains the best
terminal-LoRA two-layer configuration, replaces grouped recording loss with
recording-balanced window loss, and explicitly preserves unlabeled gaps. Validation
still averages every eligible window into a full-recording prediction. The shorter
three-event patience limits wasted time if it repeats the earlier failure. The
existing run `05-balanced-window-regression-only` used legacy gap concatenation and
is retained only as a comparison.
