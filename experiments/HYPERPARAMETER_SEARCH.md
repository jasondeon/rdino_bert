# Hyperparameter search

This search is intentionally limited to unresolved training settings. It fixes
the decisions already tested manually: 30-second windows with a 20-second
stride, shuffled window-level training, square-root inverse-frequency class
weights, standardized regression labels, and no audio augmentation or stride
jitter.

Optuna searches learning rate, weight decay, classification-loss weight, and the
plateau scheduler's factor and patience. The objective is the best-epoch,
full-recording validation R-squared. After screening, the strongest two
configurations are rerun on seeds 41 and 42 and ranked using their three-seed
mean (including screening seed 40).

Only one training process is allowed at a time. After five startup trials,
Optuna's median pruner can stop clearly weak trials after their third validation
epoch; promising trials continue under the normal early-stopping policy. Tune
these safeguards with `--pruning-startup-trials` and
`--pruning-warmup-epochs`.

First validate the configuration without starting GPU work:

```bash
bash scripts/run_hpo.sh --dry-run
```

Then run it from a normal terminal, without keeping Codex open:

```bash
bash scripts/run_hpo.sh
```

The default budget is 12 screening runs plus four confirmation runs. Use a
smaller initial budget if desired:

```bash
bash scripts/run_hpo.sh --n-trials 8 --top-k 2
```

The study is resumable. Repeating the same command continues the SQLite study
until its target total trial count is reached and reuses completed confirmation
runs. Do not change the search space while reusing the same study directory;
use a new `--study-name`, `--study-dir`, and `--output-root` for a different
search.

Review these files afterward:

- `experiments/hpo_existing_settings/report.md`
- `experiments/hpo_existing_settings/screening_trials.csv`
- `experiments/hpo_existing_settings/parameter_importance.json`
- `experiments/hpo_existing_settings/confirmation_runs.csv`
- `experiments/hpo_existing_settings/confirmation_summary.csv`

Each run retains its training history, recording predictions, best checkpoint,
console log, and training curves under `outputs/hpo/`. These artifacts are the
inputs for subsequent diagnosis of overfitting, modality collapse, aggregation,
gradient flow, or label/site effects.
