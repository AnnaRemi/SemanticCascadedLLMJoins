# Trummer ablation: is the gain from the cascade or from the expensive batch classifier?

Control = CasTrummer with `--calibration-budget 0`: the cheap stage never rejects or accepts, every pair goes to the expensive batch classifier (same code, models, 10 reps).

Means over 10 questions (paired differences are mean ± standard error over questions).

## IMDb 10q

| Variant | Precision | Recall | F1 | Wall (s) | Expensive calls | Cost (USD) |
|---|---|---|---|---|---|---|
| Trummer baseline | 0.620 | 0.275 | 0.340 | 25.4 | 10.5 | 0.0212 |
| CasTrummer (cascade) | 0.779 | 0.708 | 0.713 | 25.7 | 2.6 | 0.0214 |
| Control (no calibration) | 0.788 | 0.725 | 0.723 | 22.3 | 4.0 | 0.0186 |

- cascade − control: F1 -0.010 ± 0.035, wall +3.4 ± 0.9 s
- control − baseline: F1 +0.383 ± 0.068, wall -3.1 ± 1.1 s

## Amazon Fashion 10q

| Variant | Precision | Recall | F1 | Wall (s) | Expensive calls | Cost (USD) |
|---|---|---|---|---|---|---|
| Trummer baseline | 0.565 | 0.473 | 0.464 | 24.0 | 9.0 | 0.0200 |
| CasTrummer (cascade) | 0.755 | 0.647 | 0.673 | 23.2 | 2.0 | 0.0193 |
| Control (no calibration) | 0.673 | 0.633 | 0.621 | 21.4 | 2.8 | 0.0178 |

- cascade − control: F1 +0.052 ± 0.026, wall +1.8 ± 1.2 s
- control − baseline: F1 +0.157 ± 0.052, wall -2.6 ± 2.9 s

