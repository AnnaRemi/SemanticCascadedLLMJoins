# Rerunning the paper experiments (IMDb + Amazon Fashion, 10q × 10 reps)

## Why

Tables 1–2 and Figure 3 of the draft come from runs made in July (IMDb 2026-07-22,
Amazon 2026-07-24). Those runs predate the Beta-credible-bound calibration described
in Sec. 4.4: the saved Amazon metrics record the old agreement-based calibration and
no credible level. The numbers in the paper therefore do not describe the code in
this repository. We need fresh runs of all four methods, same models and
hyper-parameters, and then new tables and a new Figure 3.

**Deliverable:** for each dataset, one output directory with `aggregate.csv`,
`comparison.csv` and `per_question/` (section 6), plus a `paper_outputs/` folder with
Table 1 (IMDb), Table 2 (Amazon) and Figure 3, produced by two commands (section 8) from
those two `comparison.csv` files. Following sections 2 to 8 in order reproduces the three
paper artifacts from the code in the commit you were given; nothing has to be copied by
hand except the numbers into the paper (section 9).

Both cascades changed again on 2026-10-05, after a first rerun showed them losing to their
baselines (SUQL v1 rejected every row on 3 of 10 IMDb questions). The calibration now
(a) draws the 20 labelled rows **stratified by cheap score**, (b) certifies the reject side
with a Beta credible lower bound on **recall**, as in paper Sec. 4.4, instead of the
reject precision TN/(TN+FN) that passes trivially when positives are rare, (c) **reuses**
the labelled rows' expensive answers instead of asking again, and (d) for SUQL v1 reads
**real log-probs** from Ollama's native `/api/generate` (the `/v1/completions` endpoint
ignores `logprobs`, so scores used to collapse to +-2). On Amazon, SUQL v1 no longer
calibrates against the benchmark's ground truth. Trummer V1's cheap scores stay
three-valued (YES/UNCERTAIN/NO) because they come from a batched prompt. Run the 2-rep
check of section 7b before spending a 10-rep job.

## 1. What you need

- A cluster with OAR and one GPU per job. The July runs used Kraken, **H200**
  (`gpumodel='H200'`). Use the same GPU type: wall-clock time and the dollar cost
  (GPU seconds × $3/h) are compared across methods, and they change with the GPU.
- Key-based `ssh` and `rsync` from your machine to the cluster login node.
- An Ollama binary on the cluster. The worker looks at `$OLLAMA_BIN`, then `PATH`, then
  `~/.local/ollama/bin`, `~/.local/bin`, `~/bin`, `/usr/local/bin`, `/usr/bin`.
- Space in your Ollama model store for `gemma4:e2b` and `gemma4:26b`.
- The job creates `$AKER_ROOT/.venv` and runs `pip install` on the compute node. If your
  nodes have no outbound access, create that venv beforehand from the login node.
- Locally, only for post-processing (section 8): Python 3.10+ with `pandas numpy matplotlib`.

## 2. Get the code

```bash
git clone git@github.com:AnnaRemi/SemanticCascadedLLMJoins.git
cd SemanticCascadedLLMJoins
git checkout <COMMIT>        # Anna gives you the commit; do not run from another one
```

Pre-flight checks. Run them from the repo root before submitting anything:

```bash
# 1. Target quality Q*: the worker default is what the cluster job passes to every method
grep -n "CASCADE_TARGET:-" imdb/benchmarks/shared/scripts/_aker_worker.sh \
  amazon/benchmarks/shared/scripts/_aker_worker.sh

# 2. Credible level alpha. SUQL v1 and Trummer V1 take it from different places:
grep -n 'SUQL_CREDIBLE_LEVEL", "' "imdb/approaches/project SUQL/v1/suql_engine.py" \
  "amazon/approaches/project SUQL/v1/suql_engine.py"
grep -n "credible_level: float" "imdb/approaches/project Trummer/v1/trummer_join/cascade.py" \
  "amazon/approaches/project Trummer/v1/trummer_join/cascade.py"
grep -n "credible" amazon/benchmarks/shared/scripts/run_method.py   # if present, overrides SUQL v1's alpha on Amazon

# 3. The Amazon sync script must ship the approaches/ directories (expect 4 lines)
grep -n "approaches/project" amazon/benchmarks/sync_to_aker.sh

# 3b. The stratified Beta estimator must be present in all four cascade packages (expect 4 files)
ls "imdb/approaches/project SUQL/v1/band_fit.py" "amazon/approaches/project SUQL/v1/band_fit.py" \
   "imdb/approaches/project Trummer/v1/trummer_join/band_fit.py" \
   "amazon/approaches/project Trummer/v1/trummer_join/band_fit.py"
grep -c "SUQL_GROUND_TRUTH_IDS" amazon/benchmarks/shared/scripts/run_method.py   # expect 1: only the os.environ.pop

# 4. The post-processing scripts of section 8 must be in this commit (expect both paths)
ls imdb/benchmarks/multi_model_experiments/plots/make_paper_tables.py \
   imdb/benchmarks/multi_model_experiments/plots/make_paper_style_cost_pareto.py
```

Expected in the commit you were given: check 3b lists four files and prints `1`; check 1 prints `0.8` for both datasets; check 2
prints `0.9` on every line, including the Amazon `--credible-level` default; check 3 prints
four lines; check 4 lists both files. If check 3 prints nothing, the Amazon sync cannot
work: stop and ask Anna. If check 4 reports a missing file, the commit predates the
post-processing scripts: stop and ask Anna. If check 1 or 2 shows any other value, stop and
ask Anna. **Do not "correct" Q\* to the paper's 0.9**, for the reason in 2.1.

### 2.1 Why Q\* = 0.9, α = 0.9 cannot be used with |C| = 20, and what |C| = 20 can do

A side of the band is certified only if the credible lower bound on its recall (reject side)
or precision (accept side) reaches Q\*. With 20 labelled rows, even a *perfect* cheap stage
cannot reach 0.9 at α = 0.9 (20 correct out of 20 gives a bound of 0.896). The table gives the
minimum number of correct labelled rows, with zero errors, that one side needs:

| Q\* | α = 0.90 | α = 0.85 |
|---|---|---|
| 0.90 | 21 | 18 |
| 0.85 | 14 | 11 |
| 0.80 | 10 | 8 |
| 0.75 | 8 | 6 |
| 0.70 | 6 | 5 |

So Q\* = 0.9 at α = 0.9 can never activate the calibration. The runs use
**Q\* = 0.8, α = 0.9**.

The estimator is stratified (`band_fit.py`): the pool is cut into four equal-count strata by
cheap score, the 20 labels are spread over them, each stratum gets its own Beta posterior for
its positive rate, and the bound on the whole band is obtained by Monte Carlo over those
posteriors. A simulation (60 pools, N = 200, 30% positives) found that whenever a reject
region was certified the true recall reached Q\* in 100% of trials (target: at least 90%).
The price of a *valid* bound is that it is modest: with |C| = 20 the cheap stage can decide
without verification only about 10–14% of the rows, however large the pool, because every
stratum it rejects must be backed by labels of its own. Deciding more than half of a
1000-row pool took |C| around 100 in simulation. Expect the cascades to match the baselines'
quality but to save little or nothing at N = 40 unless the cheap model is much cheaper per
call than the expensive one; the 10-rep numbers will say so plainly, and the paper text must
follow them.

## 3. Settings (paper Sec. 5.1) and where they live

| Setting | Paper | How to get it |
|---|---|---|
| Suite | 10q, both datasets | `--suite 10q` |
| Methods | `suql_baseline suql_v1 trummer_baseline trummer_v1` | default of `run_aker.sh` |
| Repetitions | 10 | `--repetitions 10` |
| Cheap / expensive model | `gemma4:e2b` / `gemma4:26b` | `--cheap-model`, `--expensive-model` |
| Target quality Q\* | 0.9 in the paper, infeasible (2.1) | **0.8**, default in the worker (check 1) |
| Credible level α | 0.9 | **0.9**, defaults in the code (check 2) |
| Calibration set \|C\| | 20 | default (`CALIBRATION_BUDGET`) |
| Batch sizes B / B′ | 8 / 32 | default (`CHEAP_BATCH_SIZE`, `EXPENSIVE_BATCH_SIZE`) |
| Parallel repetitions | not stated | default 4 (`--parallel-workers`); keep it, it affects wall time |
| Semantic dictionary | not stated | on by default; do **not** pass `--no-semantic-dict` (ablation switch) |

Do not pass `--allow-cpu`. The worker refuses to run unless Ollama is resident on the GPU.

Two things to know about the question sets:

- IMDb `10q`: 10 questions, 100 candidate movies each, 40 pass the structured filter,
  12 are answers.
- Amazon `10q` in the current repository is the **held-out** question set. Table 2 in the
  draft was built from the older nested set, which is archived and not what you will run.
  The new Amazon numbers will differ from the draft for that reason too, not only because
  of the code.

## 4. Submit

Each dataset needs **its own `AKER_ROOT`**. The default points to Anna's home directory
and, more importantly, `sync_to_aker.sh` runs `rsync --delete` into
`$AKER_ROOT/{benchmarks,data,approaches}`. IMDb and Amazon would overwrite each other if
they shared a root.

`AKER_HOST` is your ssh alias for the login node. A host name starting with `kraken` makes
the script submit with `-p "gpumodel='H200'" --project pr-daisyllm`; any other name uses
the plain `oarsub -S` path. If your OAR project is not `pr-daisyllm`, edit the `oarsub` line
in `run_aker.sh`.

IMDb:

```bash
export AKER_HOST=<your-ssh-alias>
export AKER_ROOT=/home/<you>/lab_m2_imdb
cd imdb/benchmarks
bash run_aker.sh --suite 10q --repetitions 10 \
  --cheap-model gemma4:e2b --expensive-model gemma4:26b \
  --output-name imdb_10q_10rep_beta_gemma4_e2b_26b --pull-models
```

Amazon:

```bash
export AKER_HOST=<your-ssh-alias>
export AKER_ROOT=/home/<you>/lab_m2_amazon
cd amazon/benchmarks
bash run_aker.sh --suite 10q --repetitions 10 \
  --cheap-model gemma4:e2b --expensive-model gemma4:26b \
  --output-name amazon_10q_10rep_beta_gemma4_e2b_26b --pull-models
```

What happens: the script syncs code and data to `$AKER_ROOT`, writes a wrapper under
`$AKER_ROOT/benchmarks/10q/jobs/`, submits one OAR job, and prints the output path. Note
the job id from the `oarsub` output.

- `--pull-models` is needed on the first run; without it the job stops with
  "missing model". Repeating it is harmless.
- The default walltime is `24:00:00`; change it with `--walltime HH:MM:SS`. I could not
  find a recorded duration for a complete 10q × 10-rep run, so size it generously. If a job
  hits its walltime, resubmit only the unfinished methods with
  `--methods "suql_v1 trummer_v1"` and a new `--output-name`. Section 8 takes several
  `comparison.csv` files per dataset, so list all of them there.
- Do not change the checkout while a job from that root is running, and do not re-run
  `run_aker.sh` for that root with different code: the sync replaces what the job reads.

## 5. Monitor

```bash
ssh $AKER_HOST "oarstat -u"
ssh $AKER_HOST "ls -t $AKER_ROOT/benchmarks/10q/logs | head"
ssh $AKER_HOST "tail -f $AKER_ROOT/benchmarks/10q/logs/benchmark_<jobid>_<stamp>.console.log"
```

At the top of the console log check these lines: `Models: cheap=gemma4:e2b expensive=gemma4:26b`,
`Verified GPU residency for ...` for both models, and `Semantic dict: enabled` (printed only
by newer versions of the worker). Error text
goes to `oar_<jobid>.err` and `ollama_<jobid>_<stamp>.log` in the same `logs/` directory.

## 6. Collect

```bash
# IMDb
rsync -av $AKER_HOST:$AKER_ROOT/benchmarks/10q/outputs/<output-name>/ \
  imdb/benchmarks/10q/outputs/<output-name>/
# Amazon (use the Amazon AKER_ROOT)
rsync -av $AKER_HOST:$AKER_ROOT/benchmarks/10q/outputs/<output-name>/ \
  amazon/benchmarks/10q/outputs/<output-name>/
```

No `--delete`. For 10 repetitions the run keeps `aggregate.csv`, `comparison.csv`,
`plots/`, and per question and method a `run_metrics_repetitions.csv`, `run_metrics.json`
and logs. For the SUQL methods it also keeps `engine_metrics.json`, but only for the last
repetition.

## 7. Check the run before using any number

1. `comparison.csv` has 40 rows (10 questions × 4 methods), and the `repetitions`
   column is 10 on every row (2 for the check run of 7b).
2. No method row is missing or all zeros, and `stderr.log` files are not full of tracebacks.
3. Calibration. Both cascades now record, **per repetition**, in
   `per_question/q_XX/{suql_v1*,trummer_v1*}/run_metrics_repetitions.csv`:
   `calibration_activated` (0/1), `calibration_reject_recall_lower`,
   `calibration_accept_precision_lower`, `calibration_reused_labels`, `cheap_early_accepts`,
   `cheap_early_rejects`. `run_metrics.json` adds `calibration_activated_rate` (the share of
   repetitions that activated; the bare `calibration_activated` there is only the last
   repetition). SUQL v1's `run_metrics.json` also has `cheap_score_mode`, and
   `suql_v1*/engine_metrics.json` has `cascade_target`, which must equal the Q\* from
   pre-flight check 1. **`cheap_score_mode` must be `native_logprobs`.** `label_only` means
   Ollama returned no log-probs and every cheap score is just ±2, so SUQL v1's calibration is
   meaningless: stop and tell Anna. α is not saved per run: rely on pre-flight check 2.
4. Write down for Anna: commit hash, GPU model, job ids, `--parallel-workers`, the Q\*, α
   and |C| the run used, the activation shares from item 3 for **both** cascades, and any
   deviation from section 3.

Commands for item 3 (once per dataset, with that dataset's output folder):

```bash
OUT=imdb/benchmarks/10q/outputs/imdb_10q_10rep_beta_gemma4_e2b_26b   # or the Amazon folder
python3 - "$OUT" <<'PY'
import csv, glob, sys
for method in ("suql_v1", "trummer_v1"):
    reps = [r for f in sorted(glob.glob(f"{sys.argv[1]}/per_question/q_*/{method}*/run_metrics_repetitions.csv"))
            for r in csv.DictReader(open(f))]
    on = sum(int(r["calibration_activated"]) for r in reps)
    rejected = sum(float(r["cheap_early_rejects"]) for r in reps) / max(1, len(reps))
    print(f"{method}: calibration active in {on}/{len(reps)} repetitions; mean rows rejected by the cheap stage {rejected:.1f}")
PY
grep -ho '"cheap_score_mode": [^,}]*' "$OUT"/per_question/q_*/suql_v1*/engine_metrics.json | sort | uniq -c   # must be native_logprobs only
grep -ho '"cascade_target": [^,}]*' "$OUT"/per_question/q_*/suql_v1*/engine_metrics.json | sort -u   # one line, = Q*
```

If a cascade is active in 0 repetitions, say so: with the recall test that is a legitimate
outcome (the cheap stage could not be certified), and its column then equals the baseline
plus the cheap pass. It does not mean the run failed, but the paper must not describe that
column as a working cascade.

### 7b. Two-repetition check before the 10-rep job

`bash run_aker.sh --suite 10q --repetitions 2 ...` (as in section 4, other `--output-name`).
Expected: items 1 to 3 hold, SUQL v1 shows `native_logprobs`, no question has F1 = 0 for a
cascade whose baseline is above 0.3 (the failure this fix addresses), and `expensive_calls`
of a cascade never exceeds the baseline's by more than the number of cheap-decided rows.

## 8. Rebuild Tables 1–2 and Figure 3

Both commands read only the two `comparison.csv` files, so the tables and the figure come
from the same numbers. Run them from the repo root, after `pip install pandas numpy matplotlib`:

```bash
python3 imdb/benchmarks/multi_model_experiments/plots/make_paper_tables.py \
  --imdb   imdb/benchmarks/10q/outputs/imdb_10q_10rep_beta_gemma4_e2b_26b/comparison.csv \
  --amazon amazon/benchmarks/10q/outputs/amazon_10q_10rep_beta_gemma4_e2b_26b/comparison.csv \
  --out-dir paper_outputs

python3 imdb/benchmarks/multi_model_experiments/plots/make_paper_style_cost_pareto.py \
  --imdb-aggregate   paper_outputs/aggregate_imdb.csv \
  --amazon-aggregate paper_outputs/aggregate_amazon.csv \
  --out-dir paper_outputs
```

If you used other `--output-name`s, change the paths. If a run was split over several jobs
(section 4), list all of its `comparison.csv` files after `--imdb` or `--amazon`; the
methods must not overlap, and the script stops if a question/method pair appears twice.
Do not run the Pareto script without arguments: its defaults are the July runs.

What the first command does: each cell is the mean ± sample standard deviation (n − 1)
across the 10 questions, after averaging the repetitions of each question; the dollar cost
is `(cheap_seconds + expensive_seconds) × $3/3600`. It prints both tables and stops with a
message if a method name is not in `METHOD_ALIASES` (in `make_paper_style_cost_pareto.py`).
It prints `WARNING:` on stderr if a method has other than 10 questions or 10 repetitions,
or an F1 of zero on every question. A healthy run prints no warning.

| File in `paper_outputs/` | What it is |
|---|---|
| `table_imdb.md`, `table_imdb_rows.tex` | Table 1 (IMDb): Precision, Recall, F1, Wall time, LLM calls |
| `table_amazon.md`, `table_amazon_rows.tex` | Table 2 (Amazon): the same rows plus `$-cost` |
| `cost_f1_pareto_both_datasets.png` | Figure 3 |
| `imdb_cost_f1_pareto.png`, `amazon_cost_f1_pareto.png` | the per-dataset versions of Figure 3 |
| `aggregate_imdb.csv`, `aggregate_amazon.csv` | the per-method means that Figure 3 plots |

The `*_rows.tex` files use the draft's formatting (`$\pm$`, `\textbf` for the best value of a
row, `\underline` for the second best) and paste between `\midrule` and `\bottomrule` of the
draft's tables. The draft's Table 1 has no `$-cost` row, so neither does `table_imdb.md`.
The "Relative change" lines printed under each table (CasTrummer vs SUQL, CasSuql vs SUQL,
CasTrummer vs Trummer: LLM calls, wall time, F1) are the percentages quoted in the abstract
and Sec. 5.2.

### Optional check that the scripts reproduce the draft

If you have the July output folders (Anna can send them), the same two commands with the
July paths reproduce the draft: Figure 3 is pixel-identical, and every cell of Tables 1–2
matches except the standard deviation of IMDb SUQL wall time, which comes out as 12.27
where the draft prints 12.28 (the draft's value was computed from per-question times rounded
to two decimals; the unrounded value is 12.2749). The July Amazon folder uses an older
per-repetition `comparison.csv` layout, which the script also reads.

```bash
python3 imdb/benchmarks/multi_model_experiments/plots/make_paper_tables.py \
  --imdb   imdb/benchmarks/10q/outputs/heldout_diverse_10q_10rep_gemma4_e2b_26b_kraken_20260722/comparison.csv \
  --amazon amazon/archive_generalized_pipeline_results/10q_nested_archive/amazon_fashion_10q_10rep_gemma4_e2b_gemma4_26b_20260724_223605/comparison.csv \
  --out-dir paper_outputs_july_check

python3 imdb/benchmarks/multi_model_experiments/plots/make_paper_style_cost_pareto.py \
  --imdb-aggregate   paper_outputs_july_check/aggregate_imdb.csv \
  --amazon-aggregate paper_outputs_july_check/aggregate_amazon.csv \
  --out-dir paper_outputs_july_check
```

## 9. Hand back

Send Anna the `paper_outputs/` folder, the two `comparison.csv` files, and the notes from
section 7 item 4. Anna then updates the paper:

- Tables 1–2: paste the `*_rows.tex` rows. Figure 3: copy `cost_f1_pareto_both_datasets.png`
  into the paper's figures (the scripts do not touch the paper folder).
- Every percentage in the abstract, introduction and Sec. 5.2: take it from the "Relative
  change" lines, and re-check each ranking claim ("dominates", "cheapest") against the new
  Figure 3.
- Sec. 4.4 and 5.1: the draft says Q\* = 0.9; the runs use the Q\*, α and |C| recorded in
  section 7 (see 2.1). Name the GPU model. The draft's Amazon pool statistics (50 products
  satisfy σ_S, 15 answers) describe the older nested question set, not the held-out one
  this run uses.


## 10. Optional: model-pair latency probe and scaling study (IMDb only)

**Probe.** `imdb/benchmarks/shared/scripts/latency_probe.py` (run on a GPU node by
`_aker_probe.sh`) measures, per model and with the benchmark's own scoring and answer code:
seconds per cheap scoring call and per expensive answer call, whether Ollama returns log-probs
(`/v1/completions` never does; the native `/api/generate` does), how many distinct cheap
scores come back, and ROC-AUC of the cheap score against the ground truth. Submit it like a
benchmark job (see `sync_to_aker.sh` first), with `PROBE_MODELS="a,b,c"` exported in the OAR
wrapper and output under `$AKER_ROOT/benchmarks/probes/<name>/probe.json`. First results
(H200, 2026-10-05, 120 rows, single stream):

| Model | s per scoring call | s per answer call | AUC vs ground truth |
|---|---|---|---|
| gemma4:e2b | 0.42 | 0.59 | 0.848 |
| gemma4:26b | 0.47 | 0.65 | 0.896 |
| gemma4:31b | 0.71 | 0.79 | 0.878 |
| llama3.1:8b | 0.23 | 0.39 | 0.812 |
| llama3.1:70b | 0.47 | 0.78 | 0.881 |
| qwen3.5:2b | 0.36 | 0.67 | 0.829 |
| qwen3.5:27b | 0.84 | 1.00 | 0.905 |

A cascade can only save time if the cheap *scoring* call is much cheaper than the expensive
*answer* call. The gemma4 pair the paper uses has a ratio of 0.42 / 0.65 = 0.65; llama3.1
8b to 70b has 0.23 / 0.78 = 0.30, the largest gap measured.

**Scaling.** `python3 imdb/benchmarks/build_scaling_suites.py` writes `scale_x1`, `scale_x2`
and `scale_x4` (questions 2, 4, 7, 8, 10 of 10q, with 40, 80 and 160 candidates and 30%
positives; the corpus has only 3,209 movies, so 160 is the ceiling). Run them with
`bash run_aker.sh --suite scale_x4 --repetitions 2 --calibration-budget 20 --output-name ...`
and summarise with `python3 imdb/benchmarks/analyze_scaling.py --tags gemma,llama`
(output names must be `scale_x<S>_<tag>_<reps>rep`).
