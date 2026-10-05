#!/usr/bin/env python3
"""Rebuild the paper's Tables 1-2 from one or more comparison.csv files, and
write the per-method aggregate.csv that make_paper_style_cost_pareto.py plots,
so the tables and Figure 3 always come from the same numbers.

Each cell is mean +- sample standard deviation (ddof=1) across the questions of
the per-question means (repetitions are averaged first). Cost is
(cheap_seconds + expensive_seconds) * $3/h unless the file has total_cost_usd.

Pass several comparison.csv files per dataset when a run was split over several
jobs (their methods must not overlap). Both the current per-question layout
(one row per question and method, with a `repetitions` column) and the older
per-repetition layout (a `repetition` column) are accepted.

    python make_paper_tables.py \
        --imdb   imdb/benchmarks/10q/outputs/<name>/comparison.csv \
        --amazon amazon/benchmarks/10q/outputs/<name>/comparison.csv \
        --out-dir paper_outputs
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from make_paper_style_cost_pareto import ACCELERATOR_USD_PER_HOUR, METHOD_ALIASES, METHOD_ORDER

# (column, label, decimals, higher_is_better)
METRICS = [
    ("precision", "Precision", 3, True),
    ("recall", "Recall", 3, True),
    ("f1", "F1", 3, True),
    ("wall_seconds", "Wall time (s)", 2, False),
    ("llm_calls", "LLM calls", 2, False),
    ("total_cost_usd", "$-cost", 4, False),
]
# The draft's Table 1 (IMDb) has no $-cost row; Table 2 (Amazon) does.
TABLES = {
    "imdb": {"title": "IMDb 10q", "metrics": METRICS[:-1]},
    "amazon": {"title": "Amazon Fashion 10q", "metrics": METRICS},
}
AGGREGATE_COLUMNS = [
    "precision", "recall", "f1", "wall_seconds", "llm_calls", "cheap_calls",
    "expensive_calls", "cheap_seconds", "expensive_seconds", "total_cost_usd",
]
# (method, baseline) pairs behind the relative changes quoted in the abstract and Sec. 5.2.
COMPARISONS = [("CasTrummer", "SUQL"), ("CasSuql", "SUQL"), ("CasTrummer", "Trummer")]


def warn(message: str) -> None:
    print(f"WARNING: {message}", file=sys.stderr)


def load(paths: list[Path], dataset: str) -> pd.DataFrame:
    frame = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    key = "implementation" if "implementation" in frame.columns else "method"
    frame["method_label"] = frame[key].map(METHOD_ALIASES)
    unknown = sorted(frame.loc[frame["method_label"].isna(), key].unique())
    if unknown:
        raise SystemExit(
            f"{dataset}: unknown method name(s) {unknown}; add them to METHOD_ALIASES "
            "in make_paper_style_cost_pareto.py"
        )
    frame["implementation"] = frame[key]
    if "total_cost_usd" not in frame.columns:
        frame["total_cost_usd"] = (
            (frame["cheap_seconds"] + frame["expensive_seconds"]) * ACCELERATOR_USD_PER_HOUR / 3600
        )
    if "repetition" not in frame.columns:
        duplicated = frame.duplicated(["question", "method_label"], keep=False)
        if duplicated.any():
            pairs = frame.loc[duplicated, ["question", "method_label"]].drop_duplicates().to_dict("records")
            raise SystemExit(f"{dataset}: the same question/method appears more than once: {pairs[:5]}")
    return frame


def per_question(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [c for c in AGGREGATE_COLUMNS if c in frame.columns]
    return frame.groupby(["method_label", "question"], observed=True)[numeric].mean().reset_index()


def check_run(frame: pd.DataFrame, per_q: pd.DataFrame, dataset: str, expect_questions: int, expect_reps: int) -> None:
    missing = [m for m in METHOD_ORDER if m not in set(per_q["method_label"])]
    if missing:
        warn(f"{dataset}: no rows for {missing}")
    for method, group in per_q.groupby("method_label", observed=True):
        if len(group) != expect_questions:
            warn(f"{dataset}/{method}: {len(group)} questions, expected {expect_questions}")
        if (group["f1"] == 0).all():
            warn(f"{dataset}/{method}: F1 is zero on every question")
    if "repetitions" in frame.columns:
        reps = frame.groupby("method_label", observed=True)["repetitions"].agg(["min", "max"])
        for method, row in reps.iterrows():
            if row["min"] != expect_reps or row["max"] != expect_reps:
                warn(f"{dataset}/{method}: repetitions {row['min']}..{row['max']}, expected {expect_reps}")
    elif "repetition" in frame.columns:
        reps = frame.groupby(["method_label", "question"], observed=True)["repetition"].nunique()
        for method, counts in reps.groupby(level=0, observed=True):
            if (counts != expect_reps).any():
                warn(f"{dataset}/{method}: repetitions per question {counts.min()}..{counts.max()}, expected {expect_reps}")


def summarise(per_q: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = per_q.groupby("method_label", observed=True)
    mean = grouped.mean(numeric_only=True).reindex(METHOD_ORDER)
    std = grouped.std(numeric_only=True, ddof=1).reindex(METHOD_ORDER)
    return mean, std


def rank_marks(means: pd.Series, higher_is_better: bool) -> dict[str, str]:
    """'best' for the top method of a row, 'second' for the runner-up."""
    ordered = means.dropna().sort_values(ascending=not higher_is_better).index.tolist()
    marks = {}
    if ordered:
        marks[ordered[0]] = "best"
    if len(ordered) > 1:
        marks[ordered[1]] = "second"
    return marks


def render(mean: pd.DataFrame, std: pd.DataFrame, metrics: list) -> tuple[str, str]:
    header = ["Metric", "SUQL", "CasSuql", "Trummer", "CasTrummer"]
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    tex = []
    for column, label, decimals, higher in metrics:
        marks = rank_marks(mean[column], higher)
        md_cells, tex_cells = [], []
        for method in METHOD_ORDER:
            if pd.isna(mean.loc[method, column]):
                md_cells.append("n/a")
                tex_cells.append("n/a")
                continue
            m, s = mean.loc[method, column], std.loc[method, column]
            md_cells.append(f"{m:.{decimals}f} ± {s:.{decimals}f}")
            cell = f"{m:.{decimals}f} $\\pm$ {s:.{decimals}f}"
            mark = marks.get(method)
            tex_cells.append(f"\\textbf{{{cell}}}" if mark == "best" else f"\\underline{{{cell}}}" if mark == "second" else cell)
        md.append(f"| {label} | " + " | ".join(md_cells) + " |")
        tex.append(f"{label.replace('$', chr(92) + '$'):<14}& " + " & ".join(tex_cells) + " \\\\")
    return "\n".join(md), "\n".join(tex)


def relative_changes(mean: pd.DataFrame) -> str:
    lines = []
    for method, base in COMPARISONS:
        if mean.loc[[method, base]].isna().any(axis=None):
            continue
        parts = []
        for column, label in (("llm_calls", "LLM calls"), ("wall_seconds", "wall time"), ("f1", "F1")):
            change = (mean.loc[method, column] / mean.loc[base, column] - 1) * 100
            parts.append(f"{label} {change:+.1f}%")
        lines.append(f"  {method} vs {base}: " + ", ".join(parts))
    return "\n".join(lines)


def build(dataset: str, paths: list[Path], out_dir: Path, expect_questions: int, expect_reps: int) -> None:
    spec = TABLES[dataset]
    frame = load(paths, dataset)
    per_q = per_question(frame)
    check_run(frame, per_q, dataset, expect_questions, expect_reps)
    mean, std = summarise(per_q)
    markdown, latex = render(mean, std, spec["metrics"])

    implementation = frame.groupby("method_label", observed=True)["implementation"].first()
    aggregate = mean[[c for c in AGGREGATE_COLUMNS if c in mean.columns]].copy()
    aggregate.insert(0, "implementation", implementation.reindex(aggregate.index))
    aggregate = aggregate.dropna(subset=["f1"]).reset_index(drop=True)
    aggregate.to_csv(out_dir / f"aggregate_{dataset}.csv", index=False)
    (out_dir / f"table_{dataset}.md").write_text(markdown + "\n")
    (out_dir / f"table_{dataset}_rows.tex").write_text(latex + "\n")

    n_questions = per_q["question"].nunique()
    print(f"\n{spec['title']} (mean ± std over {n_questions} questions)\n")
    print(markdown)
    print("\nRelative change:")
    print(relative_changes(mean))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--imdb", nargs="+", type=Path, required=True, metavar="COMPARISON_CSV")
    parser.add_argument("--amazon", nargs="+", type=Path, required=True, metavar="COMPARISON_CSV")
    parser.add_argument("--out-dir", type=Path, default=Path("paper_outputs"))
    parser.add_argument("--expect-questions", type=int, default=10)
    parser.add_argument("--expect-repetitions", type=int, default=10)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for dataset, paths in (("imdb", args.imdb), ("amazon", args.amazon)):
        build(dataset, paths, args.out_dir, args.expect_questions, args.expect_repetitions)
    print(f"\nWrote to {args.out_dir}/: aggregate_*.csv, table_*.md, table_*_rows.tex")


if __name__ == "__main__":
    main()
