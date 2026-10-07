#!/usr/bin/env python3
"""Cost-F1 Pareto fronts styled like the EDBT paper's Figure 3
(figs/imdb10q/04_best_solution.png: bubble-per-method, marker area
proportional to mean LLM calls, dashed frontier through the non-dominated
methods, log-scaled axis when costs span more than a decade) -- but with
cost on the x-axis instead of wall time, since that is the headline metric
this set of figures is about.

By default it reads the July 10q/10rep, single (Gemma) model-pair runs the
draft's Table 1 and Table 2 were built from:
  - imdb/benchmarks/10q/outputs/heldout_diverse_10q_10rep_gemma4_e2b_26b_kraken_20260722/
  - amazon/archive_generalized_pipeline_results/10q_nested_archive/
      amazon_fashion_10q_10rep_gemma4_e2b_gemma4_26b_20260724_223605/
(Both aggregate.csv files reproduce the draft's tables to 3 decimal places.)

For a new run, pass --imdb-aggregate / --amazon-aggregate. The aggregate_*.csv
files written by make_paper_tables.py are the intended input, so the figure and
the tables come from the same numbers. --out-dir chooses where the PNGs go
(default: next to this script).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]  # .../lab m2

IMDB_AGGREGATE = ROOT / "imdb/benchmarks/10q/outputs/heldout_diverse_10q_10rep_gemma4_e2b_26b_kraken_20260722/aggregate.csv"
AMAZON_AGGREGATE = ROOT / "amazon/archive_generalized_pipeline_results/10q_nested_archive/amazon_fashion_10q_10rep_gemma4_e2b_gemma4_26b_20260724_223605/aggregate.csv"

ACCELERATOR_USD_PER_HOUR = 3.0

# Match the paper's own method names (Suql / CasSuql / Trummer / CasTrummer).
METHOD_ALIASES = {
    "suql_baseline": "SUQL",
    "suql_v1_two_level_cascade": "CasSuql",
    "suql_v1": "CasSuql",
    "trummer_baseline_adaptive_block_join": "Trummer",
    "trummer_baseline": "Trummer",
    "trummer_v1_structured_two_level_cascade": "CasTrummer",
    "trummer_v1": "CasTrummer",
}
METHOD_ORDER = ["SUQL", "CasSuql", "Trummer", "CasTrummer"]
METHOD_COLORS = {
    "SUQL": "#4C78A8",
    "CasSuql": "#F2A72C",
    "Trummer": "#54A24B",
    "CasTrummer": "#E4572E",
}
DATASET_MARKERS = {"IMDb 10q": "o", "Amazon Fashion 10q": "^"}

# Multiplies every font size (and the legend marker size). 1.0 reproduces the
# original figures; ~2 keeps the text readable once the figure is shrunk to a
# single column of the paper (set with --font-scale).
FONT_SCALE = 1.0


def apply_font_scale(scale: float) -> None:
    global FONT_SCALE
    FONT_SCALE = scale
    base = plt.rcParams["font.size"]
    plt.rcParams.update({
        "font.size": base * scale,
        "axes.titlesize": plt.rcParams["axes.titlesize"] if isinstance(plt.rcParams["axes.titlesize"], str) else base * scale,
    })
    # rcParams like "large"/"medium" scale with font.size, so only font.size is needed.


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    key = "implementation" if "implementation" in frame.columns else "method"
    frame["method_label"] = frame[key].map(METHOD_ALIASES)
    if "total_cost_usd" not in frame.columns:
        usd_per_second = ACCELERATOR_USD_PER_HOUR / 3600
        frame["total_cost_usd"] = (frame["cheap_seconds"] + frame["expensive_seconds"]) * usd_per_second
    frame["method_label"] = pd.Categorical(frame["method_label"], METHOD_ORDER, ordered=True)
    return frame.sort_values("method_label").reset_index(drop=True)


def pareto_frontier(costs: np.ndarray, f1s: np.ndarray) -> list[int]:
    """Indices of points no other point dominates (lower-or-equal cost AND
    higher-or-equal F1, strictly better in at least one), sorted by cost."""
    is_pareto = [
        not any(
            j != i and costs[j] <= costs[i] and f1s[j] >= f1s[i] and (costs[j] < costs[i] or f1s[j] > f1s[i])
            for j in range(len(costs))
        )
        for i in range(len(costs))
    ]
    return sorted((i for i, keep in enumerate(is_pareto) if keep), key=lambda i: costs[i])


def _use_log(costs: np.ndarray) -> bool:
    positive = costs[costs > 0]
    return len(positive) > 0 and positive.max() / positive.min() >= 10


def _format_cost_axis(ax, use_log: bool) -> str:
    if use_log:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:g}"))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        return "Estimated cost per question (USD, log scale; lower is better)"
    return "Estimated cost per question (USD; lower is better)"


def add_footnote(fig) -> None:
    """Footnote under the axes; wrapped onto two lines when fonts are enlarged."""
    note = f"Marker area ∝ mean LLM calls per question. Compute estimate uses ${ACCELERATOR_USD_PER_HOUR:.2f}/accelerator-hour."
    if FONT_SCALE > 1.4:
        note = note.replace(". Compute", ".\nCompute")
        fig.text(.5, .008, note, ha="center", va="bottom", fontsize=8 * FONT_SCALE)
        fig.tight_layout(rect=(0, .11, 1, 1))
    else:
        fig.text(.5, .01, note, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .05, 1, 1))


def bubble_size(calls: np.ndarray) -> np.ndarray:
    return 180 + 35 * calls


def plot_single_dataset(frame: pd.DataFrame, dataset_label: str, path: Path) -> None:
    costs = frame["total_cost_usd"].to_numpy(float)
    f1s = frame["f1"].to_numpy(float)
    calls = frame["llm_calls"].to_numpy(float)
    frontier = pareto_frontier(costs, f1s)

    fig, ax = plt.subplots(figsize=(8.5, 7))
    for i, row in frame.iterrows():
        ax.scatter(
            row.total_cost_usd, row.f1, s=bubble_size(np.array([row.llm_calls]))[0],
            color=METHOD_COLORS[row.method_label], alpha=.85, zorder=3,
            edgecolor="black" if i in frontier else "white", linewidth=1.4,
        )
    if len(frontier) > 1:
        ax.plot(costs[frontier], f1s[frontier], "--", color="#333333", alpha=.7, zorder=2, label="Pareto frontier")

    use_log = _use_log(costs)
    xlabel = _format_cost_axis(ax, use_log)
    ax.margins(x=0.22, y=0.12)
    ax.set(title=f"{dataset_label}: cost–F1 Pareto frontier", xlabel=xlabel,
           ylabel="F1 (higher is better)", ylim=(-.03, 1.05))
    ax.grid(alpha=.25, which="both" if use_log else "major")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=METHOD_COLORS[m],
                   markeredgecolor="black", markersize=11 * FONT_SCALE, label=m)
        for m in METHOD_ORDER
    ]
    if len(frontier) > 1:
        handles.append(plt.Line2D([0], [0], linestyle="--", color="#333333", alpha=.7, label="Pareto frontier"))
    ax.legend(handles=handles, loc="lower right", fontsize=9.5 * FONT_SCALE * (0.8 if FONT_SCALE > 1.4 else 1), framealpha=.9)
    add_footnote(fig)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_combined(frames: dict[str, pd.DataFrame], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    all_costs = []
    for dataset_label, frame in frames.items():
        costs = frame["total_cost_usd"].to_numpy(float)
        f1s = frame["f1"].to_numpy(float)
        calls = frame["llm_calls"].to_numpy(float)
        all_costs.append(costs)
        frontier = pareto_frontier(costs, f1s)
        for i, row in frame.reset_index(drop=True).iterrows():
            ax.scatter(
                row.total_cost_usd, row.f1, s=bubble_size(np.array([row.llm_calls]))[0],
                color=METHOD_COLORS[row.method_label], marker=DATASET_MARKERS[dataset_label],
                alpha=.85, zorder=3, edgecolor="black" if i in frontier else "white", linewidth=1.4,
            )
        if len(frontier) > 1:
            ax.plot(costs[frontier], f1s[frontier], "--", color="#333333", alpha=.7, zorder=2)

    use_log = _use_log(np.concatenate(all_costs))
    xlabel = _format_cost_axis(ax, use_log)
    ax.margins(x=0.22, y=0.12)
    ax.set(title=("Cost–F1 Pareto frontier (10q, Gemma pair)" if FONT_SCALE > 1.4
                  else "Cost–F1 Pareto frontier, both datasets (10q, Gemma pair)"), xlabel=xlabel,
           ylabel="F1 (higher is better)", ylim=(-.03, 1.05))
    ax.grid(alpha=.25, which="both" if use_log else "major")

    method_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=METHOD_COLORS[m],
                   markeredgecolor="black", markersize=11 * FONT_SCALE, label=m)
        for m in METHOD_ORDER
    ]
    dataset_handles = [
        plt.Line2D([0], [0], marker=marker, color="w", markerfacecolor="#999999",
                   markeredgecolor="black", markersize=11 * FONT_SCALE, label=label)
        for label, marker in DATASET_MARKERS.items()
    ]
    frontier_handle = [plt.Line2D([0], [0], linestyle="--", color="#333333", alpha=.7, label="Pareto frontier (per dataset)")]
    ax.legend(handles=method_handles + dataset_handles + frontier_handle, loc="lower right",
              fontsize=9 * FONT_SCALE * (0.8 if FONT_SCALE > 1.4 else 1), framealpha=.9,
              ncol=2 if FONT_SCALE > 1.4 else 1)
    add_footnote(fig)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cost-F1 Pareto plots in the style of the paper's Figure 3.")
    parser.add_argument("--imdb-aggregate", type=Path, default=IMDB_AGGREGATE)
    parser.add_argument("--amazon-aggregate", type=Path, default=AMAZON_AGGREGATE)
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--font-scale", type=float, default=1.0,
                        help="multiply all font sizes (default 1.0; ~2 for a single-column paper figure)")
    args = parser.parse_args()
    if args.font_scale != 1.0:
        apply_font_scale(args.font_scale)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "IMDb 10q": load(args.imdb_aggregate),
        "Amazon Fashion 10q": load(args.amazon_aggregate),
    }
    plot_single_dataset(frames["IMDb 10q"], "IMDb 10q", args.out_dir / "imdb_cost_f1_pareto.png")
    plot_single_dataset(frames["Amazon Fashion 10q"], "Amazon Fashion 10q", args.out_dir / "amazon_cost_f1_pareto.png")
    plot_combined(frames, args.out_dir / "cost_f1_pareto_both_datasets.png")
    print("Wrote:")
    for name in ("imdb_cost_f1_pareto.png", "amazon_cost_f1_pareto.png", "cost_f1_pareto_both_datasets.png"):
        print(" -", args.out_dir / name)


if __name__ == "__main__":
    main()
