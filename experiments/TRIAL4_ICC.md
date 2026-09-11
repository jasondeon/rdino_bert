# Trial 4 ICC run

Validation now reports ICC(2,1): two-way random-effects, absolute agreement,
single measurement. It is computed from the full-recording observed and
predicted regression scores. Early stopping and checkpoint selection remain
based on recording-level R2.

This standalone run reproduces the trial 4 configuration with a new seed 44.
Inspect the command without launching training:

```bash
./scripts/run_trial4_icc.sh --dry-run
```

Run it independently of Codex:

```bash
./scripts/run_trial4_icc.sh
```

Set a different seed or output directory if needed:

```bash
SEED=45 OUTPUT_DIR=outputs/trial4-icc-seed45 ./scripts/run_trial4_icc.sh
```
