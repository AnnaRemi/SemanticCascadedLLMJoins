# Run notes (runbook section 7, item 4)

- Date: 2026-10-05. Git HEAD: e9d94af; working tree has 164 uncommitted paths (all fixes below are uncommitted).
- GPU: NVIDIA H200 (kraken, `gpumodel='H200'`), one GPU per job.
- Main jobs: IMDb 207618 (68 min), Amazon 207619 (72 min). Control jobs (Trummer v1 only, `--calibration-budget 0`): IMDb 207620, Amazon 207621.
- Settings: suite 10q, 10 repetitions, gemma4:e2b (cheap) / gemma4:26b (expensive), Q*=0.8 (paper text says 0.9; infeasible with |C|=20), alpha=0.9, |C|=20, B/B'=8/32, 4 parallel workers, semantic dictionary on.
- Calibration: 4 strata by cheap score, reject side certified by a Beta credible lower bound on recall, accept side on precision (stratified posterior, 2000 Monte-Carlo draws); calibration labels reused as final answers; SUQL v1 reads native Ollama log-probs; Trummer v1 splits and retries failed expensive batches; Amazon calibrates on expensive-model labels (no ground-truth leak).
- Section 7 checks: both comparison.csv have 40 rows, repetitions = 10; SUQL v1 cheap_score_mode = native_logprobs on all 20 questions; certified recall bound >= 0.8 in every repetition that rejected rows (IMDb 34/34, Amazon 77/77); expensive_undecided = 0 for Trummer v1 everywhere.
- Activation (repetitions with a certified band): IMDb SUQL v1 34/100, Trummer v1 18/100; Amazon SUQL v1 77/100, Trummer v1 7/100. Mean rows decided by the cheap stage without verification: IMDb 2.0 / 1.4 of 40; Amazon 7.4 / 0.9 of ~49.
- Amazon 10q here is the held-out question set; the draft's Table 2 used the older nested set.
