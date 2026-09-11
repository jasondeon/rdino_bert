# Lightweight WavLM layer mixer

This experiment tests whether supervised but strongly constrained aggregation of
all frozen WavLM Base+ depths improves over the single training-CV-selected layer.
It operates entirely on the cached recording features from the layer audit; WavLM
is never loaded or updated.

The model has separate 13-way softmax weights for temporal means and temporal
standard deviations. It concatenates the two 768-dimensional mixtures and applies
LayerNorm, dropout, and a linear standardized-regression head.

Five subject-grouped training folds determine early-stopping duration. For each of
three seeds, the median best fold epoch is used to retrain on every training
recording. External validation is evaluated once and never controls training. The
three final predictions are averaged.

Run independently of Codex:

```bash
./scripts/run_wavlm_layer_mixer.sh
```

This learner is small enough for CPU if desired:

```bash
DEVICE=cpu ./scripts/run_wavlm_layer_mixer.sh
```

Results are written to `outputs/wavlm-base-plus-layer-mixer/`. The report includes
training OOF and external-validation metrics, paired comparisons against the mean
baseline and fixed layer-10 probe, and layer weights across seeds. Exact weights
should not be overinterpreted because adjacent WavLM layers are correlated.
