#!/usr/bin/env python3
"""Evaluate benchmark outputs and create aggregate and per-question plots."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


LABELS = {
    "suql_baseline": "SUQL baseline",
    "suql_v1_two_level_cascade": "CasSuql",
    "trummer_baseline_adaptive_block_join": "Trummer baseline",
    "trummer_v1_structured_two_level_cascade": "CasTrummer",
}
COLORS = {"precision": "#4c78a8", "recall": "#f58518", "f1": "#6f5bd3"}
DEFAULT_ACCELERATOR_USD_PER_HOUR = 3.0


def quality(run: dict, truth: set[str]) -> dict[str, float]:
    found = set(run.get("found_movie_ids", []))
    tp, fp, fn = len(found & truth), len(found - truth), len(truth - found)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def annotated_bars(ax, bars, fmt="{:.2f}") -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, value, fmt.format(value),
                ha="center", va="bottom", fontsize=9)


def plot_quality(frame: pd.DataFrame, path: Path, title: str) -> None:
    methods = list(frame["method"]); x = np.arange(len(methods)); width = .24
    fig, ax = plt.subplots(figsize=(max(9, len(methods)*1.8), 6))
    for offset, metric in zip((-width, 0, width), ("precision", "recall", "f1")):
        bars = ax.bar(x+offset, frame[metric], width, label=metric.title(), color=COLORS[metric])
        annotated_bars(ax, bars, "{:.3f}")
    ax.set(title=f"{title}: precision, recall, and F1", ylabel="Quality score (higher is better)", ylim=(0, 1.12))
    ax.set_xticks(x, methods, rotation=12, ha="right"); ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def annotate_stack_shares(ax, x, cheap, expensive, min_pct: float = 4.0) -> None:
    """Label each stacked segment with its share of that bar's total.

    Skipped below `min_pct` since a sliver too thin to see is also too thin
    to hold legible text -- the total-per-bar label above still accounts for it.
    Accepts either a pandas Series or a numpy array for `cheap`/`expensive`.
    """
    cheap = np.asarray(cheap, dtype=float); expensive = np.asarray(expensive, dtype=float)
    totals = cheap + expensive
    for i, total in enumerate(totals):
        if total <= 0:
            continue
        c, e = cheap[i], expensive[i]
        if c > 0 and (c / total * 100) >= min_pct:
            ax.text(x[i], c / 2, f"{c/total*100:.0f}%", ha="center", va="center",
                     color="white", fontsize=9, fontweight="bold", zorder=4)
        if e > 0 and (e / total * 100) >= min_pct:
            ax.text(x[i], c + e / 2, f"{e/total*100:.0f}%", ha="center", va="center",
                     color="white", fontsize=9, fontweight="bold", zorder=4)


def plot_stacked(frame: pd.DataFrame, path: Path, title: str, kind: str) -> None:
    methods = list(frame["method"]); x = np.arange(len(methods))
    if kind == "time":
        cheap, expensive = frame["cheap_seconds"], frame["expensive_seconds"]
        ylabel, suffix = "Mean model time (seconds; lower is better)", "s"
    else:
        cheap, expensive = frame["cheap_calls"], frame["expensive_calls"]
        ylabel, suffix = "Mean LLM calls (lower is better)", " calls"
    fig, ax = plt.subplots(figsize=(max(9, len(methods)*1.8), 6))
    ax.bar(x, cheap, label="Cheap model", color="#4c9f70", zorder=3)
    ax.bar(x, expensive, bottom=cheap, label="Expensive model", color="#d95f02", zorder=3)
    totals = cheap + expensive
    for i, total in enumerate(totals): ax.text(i, total, f"total {total:.1f}{suffix}", ha="center", va="bottom")
    annotate_stack_shares(ax, x, cheap, expensive)
    ax.set(title=f"{title}: cheap vs expensive model {kind}", ylabel=ylabel)
    ax.set_xticks(x, methods, rotation=12, ha="right"); ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_tradeoff(frame: pd.DataFrame, path: Path, title: str) -> None:
    wall = frame["wall_seconds"].to_numpy(float); recall = frame["recall"].to_numpy(float)
    calls = frame["llm_calls"].to_numpy(float)
    wall_score = 1 - wall / max(wall.max(), 1e-9)
    call_score = 1 - calls / max(calls.max(), 1e-9)
    score = .6*recall + .2*wall_score + .2*call_score
    best = int(np.argmax(score))
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for i, row in frame.reset_index(drop=True).iterrows():
        ax.scatter(row.wall_seconds, row.recall, s=180+35*row.llm_calls, alpha=.8,
                   edgecolors="black" if i == best else "white", linewidths=3 if i == best else 1,
                   label=row.method)
        ax.annotate(f"{row.method}\n{row.wall_seconds:.1f}s, {row.llm_calls:.1f} calls",
                    (row.wall_seconds, row.recall), xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax.set(title=f"{title}: quality/time/call tradeoff\nBest normalized solution: {frame.iloc[best].method}",
           xlabel="Mean wall time in seconds (lower is better)", ylabel="Recall (higher is better)", ylim=(-.03, 1.08))
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def add_cost_columns(frame: pd.DataFrame, accelerator_usd_per_hour: float) -> pd.DataFrame:
    """Estimate compute cost from the model service time recorded by each run.

    These benchmarks use local Ollama models and do not record tokens, so an API
    price cannot be reconstructed.  Service-time costing is explicit,
    reproducible, and still captures different models' observed call duration.
    """
    result = frame.copy()
    usd_per_second = accelerator_usd_per_hour / 3600
    result["cheap_cost_usd"] = result["cheap_seconds"] * usd_per_second
    result["expensive_cost_usd"] = result["expensive_seconds"] * usd_per_second
    result["total_cost_usd"] = result["cheap_cost_usd"] + result["expensive_cost_usd"]
    result["cheap_cost_per_call_usd"] = np.divide(
        result["cheap_cost_usd"], result["cheap_calls"],
        out=np.zeros(len(result), dtype=float), where=result["cheap_calls"].to_numpy(float) > 0,
    )
    result["expensive_cost_per_call_usd"] = np.divide(
        result["expensive_cost_usd"], result["expensive_calls"],
        out=np.zeros(len(result), dtype=float), where=result["expensive_calls"].to_numpy(float) > 0,
    )
    return result


def plot_costs(frame: pd.DataFrame, path: Path, title: str, hourly_rate: float) -> None:
    methods = list(frame["method"]); x = np.arange(len(methods))
    cheap = frame["cheap_cost_usd"].to_numpy(float)
    expensive = frame["expensive_cost_usd"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(max(10, len(methods) * 2.1), 6.5))
    ax.bar(x, cheap, label="Cheap-model calls", color="#4c9f70", zorder=3)
    ax.bar(x, expensive, bottom=cheap, label="Expensive-model calls", color="#d95f02", zorder=3)
    annotate_stack_shares(ax, x, cheap, expensive)
    for i, row in frame.reset_index(drop=True).iterrows():
        ax.text(i, row.total_cost_usd, f"${row.total_cost_usd:.4f}", ha="center", va="bottom", fontsize=9)
        details = []
        if row.cheap_calls:
            details.append(f"cheap ${row.cheap_cost_per_call_usd:.5f}/call")
        if row.expensive_calls:
            details.append(f"exp. ${row.expensive_cost_per_call_usd:.5f}/call")
        ax.text(i, -max(frame["total_cost_usd"].max() * .08, .0001), "\n".join(details),
                ha="center", va="top", fontsize=8)
    ax.set_title(f"{title}: estimated model-call compute cost")
    ax.set_ylabel("Estimated cost per question (USD; lower is better)")
    ax.set_xticks(x, methods, rotation=12, ha="right"); ax.legend(); ax.grid(axis="y", alpha=.25)
    fig.text(.5, .01, f"Estimate = recorded model service time × ${hourly_rate:.2f}/accelerator-hour; token usage was not recorded.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .08, 1, 1)); fig.savefig(path, dpi=180); plt.close(fig)


def plot_cost_f1(frame: pd.DataFrame, path: Path, title: str, hourly_rate: float) -> None:
    costs = frame["total_cost_usd"].to_numpy(float)
    f1 = frame["f1"].to_numpy(float)
    pareto = [
        i for i in range(len(frame))
        if not any(j != i and costs[j] <= costs[i] and f1[j] >= f1[i]
                   and (costs[j] < costs[i] or f1[j] > f1[i]) for j in range(len(frame)))
    ]
    # A linear cost axis compresses every point into a sliver when one method's
    # cost is an order of magnitude apart from the rest, which is exactly the
    # regime this benchmark tends to produce (an uncascaded cheap stage can be
    # 10-30x pricier than everything else). Switch to a log axis whenever costs
    # span more than a decade so all points -- not just the outlier -- are legible.
    positive_costs = costs[costs > 0]
    use_log = len(positive_costs) > 0 and positive_costs.max() / positive_costs.min() >= 10

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for i, row in frame.reset_index(drop=True).iterrows():
        ax.scatter(row.total_cost_usd, row.f1, s=190, color=COLORS["f1"], alpha=.8,
                   edgecolor="black" if i in pareto else "white", linewidth=3 if i in pareto else 1,
                   zorder=3)
        ax.annotate(f"{row.method}\n${row.total_cost_usd:.4f}, F1 {row.f1:.3f}",
                    (row.total_cost_usd, row.f1), xytext=(7, 7), textcoords="offset points", fontsize=8)
    frontier = sorted(pareto, key=lambda i: costs[i])
    if len(frontier) > 1:
        ax.plot(costs[frontier], f1[frontier], "--", color="#333333", alpha=.6,
                 label="Cost–F1 Pareto frontier", zorder=2)
        ax.legend()
    if use_log:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:g}"))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        xlabel = "Estimated cost per question (USD, log scale; lower is better)"
    else:
        xlabel = "Estimated cost per question (USD; lower is better)"
    ax.margins(x=0.2)
    ax.set(title=f"{title}: cost vs F1", xlabel=xlabel, ylabel="F1 (higher is better)", ylim=(-.03, 1.03))
    ax.grid(alpha=.25, which="both" if use_log else "major")
    caption = f"Black outlines are non-dominated choices. Compute estimate uses ${hourly_rate:.2f}/accelerator-hour."
    if use_log:
        caption += " X-axis is log-scaled because costs span more than 10x."
    fig.text(.5, .01, caption, ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1)); fig.savefig(path, dpi=180); plt.close(fig)


def make_plots(frame: pd.DataFrame, directory: Path, title: str, accelerator_usd_per_hour: float) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    frame = add_cost_columns(frame, accelerator_usd_per_hour)
    plot_quality(frame, directory/"01_quality.png", title)
    plot_stacked(frame, directory/"02_time.png", title, "time")
    plot_stacked(frame, directory/"03_calls.png", title, "calls")
    plot_tradeoff(frame, directory/"04_best_solution.png", title)
    plot_costs(frame, directory/"05_cost_by_approach.png", title, accelerator_usd_per_hour)
    plot_cost_f1(frame, directory/"06_cost_vs_f1.png", title, accelerator_usd_per_hour)


def remove_run_artifacts(outputs_dir: Path) -> None:
    """Leave only publication-ready CSV tables and PNG plots."""
    per_question = outputs_dir / "per_question"
    if not per_question.exists():
        return
    for question_dir in per_question.iterdir():
        if not question_dir.is_dir():
            continue
        for child in question_dir.iterdir():
            if child.name == "plots":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--suite-root", type=Path, required=True); ap.add_argument("--outputs-dir", type=Path, required=True); ap.add_argument("--keep-run-artifacts", action="store_true")
    ap.add_argument("--accelerator-usd-per-hour", type=float, default=DEFAULT_ACCELERATOR_USD_PER_HOUR,
                    help="Compute-cost assumption for local Ollama model time (default: %(default)s)")
    args = ap.parse_args()
    manifest = json.loads((args.suite_root/"manifest.json").read_text()); rows=[]
    for index, item in enumerate(manifest["questions"], 1):
        q = item["directory"]; spec=json.loads((args.suite_root/"per_question"/q/"benchmark.json").read_text()); truth=set(spec["ground_truth_movie_ids"])
        for metrics_path in sorted((args.outputs_dir/"per_question"/q).glob("*/run_metrics.json")):
            run=json.loads(metrics_path.read_text()); rows.append({
                "question_index": index, "question": q, "implementation": run["implementation"],
                "method": LABELS.get(run["implementation"], run["implementation"]),
                "repetitions": run.get("repetitions",1), "wall_seconds": float(run.get("wall_seconds",0)),
                "llm_calls": float(run.get("llm_calls",0)), "cheap_calls": float(run.get("cheap_calls",0)),
                "expensive_calls": float(run.get("expensive_calls",0)), "cheap_seconds": float(run.get("cheap_seconds",0)),
                "expensive_seconds": float(run.get("expensive_seconds",run.get("wall_seconds",0))), **quality(run,truth)})
    frame=pd.DataFrame(rows)
    if frame.empty: raise SystemExit("No run_metrics.json files found")
    frame.to_csv(args.outputs_dir/"comparison.csv",index=False)
    numeric=["precision","recall","f1","wall_seconds","llm_calls","cheap_calls","expensive_calls","cheap_seconds","expensive_seconds"]
    aggregate=frame.groupby(["implementation","method"],as_index=False)[numeric].mean()
    costed_frame = add_cost_columns(frame, args.accelerator_usd_per_hour)
    costed_frame.to_csv(args.outputs_dir/"comparison.csv",index=False)
    costed_aggregate = add_cost_columns(aggregate, args.accelerator_usd_per_hour)
    costed_aggregate.to_csv(args.outputs_dir/"aggregate.csv",index=False)
    make_plots(aggregate,args.outputs_dir/"plots",f"{manifest['suite']} averaged experiment",args.accelerator_usd_per_hour)
    for q,qframe in frame.groupby("question",sort=False):
        make_plots(qframe,args.outputs_dir/"per_question"/q/"plots",f"{q}",args.accelerator_usd_per_hour)
    if not args.keep_run_artifacts:
        remove_run_artifacts(args.outputs_dir)
    print(aggregate.to_string(index=False))


if __name__ == "__main__": main()
