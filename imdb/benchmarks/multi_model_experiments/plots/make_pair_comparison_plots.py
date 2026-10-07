#!/usr/bin/env python3
"""Compare quality (precision/recall/F1) across model pairs, per dataset.

Reads each <dataset>_5q_10rep/<pair>/aggregate.csv and plots the four
methods side by side, grouped by model pair, so pairs can be compared
directly instead of only within their own subdirectory.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ACCELERATOR_USD_PER_HOUR = 3.0  # matches evaluate_and_plot.DEFAULT_ACCELERATOR_USD_PER_HOUR

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

DATASETS = {
    "imdb": ROOT / "imdb_5q_10rep",
    "amazon": ROOT / "amazon_5q_10rep",
}

PAIR_ORDER = [
    "gemma4_e2b__gemma4_26b",
    "qwen3.6_27b__qwen3.6_35b",
    "gemma4_e2b__qwen3.6_35b_crossfamily",
    "llama3.1_8b__llama3.1_70b",
]
PAIR_LABELS = {
    "gemma4_e2b__gemma4_26b": "gemma4:e2b /\ngemma4:26b",
    "qwen3.6_27b__qwen3.6_35b": "qwen3.6:27b /\nqwen3.6:35b",
    "gemma4_e2b__qwen3.6_35b_crossfamily": "gemma4:e2b /\nqwen3.6:35b (cross)",
    "llama3.1_8b__llama3.1_70b": "llama3.1:8b /\nllama3.1:70b",
}
PAIR_MARKERS = {
    "gemma4_e2b__gemma4_26b": "o",
    "qwen3.6_27b__qwen3.6_35b": "s",
    "gemma4_e2b__qwen3.6_35b_crossfamily": "^",
    "llama3.1_8b__llama3.1_70b": "D",
}

METHOD_ORDER = [
    "suql_baseline",
    "suql_v1_two_level_cascade",
    "trummer_baseline_adaptive_block_join",
    "trummer_v1_structured_two_level_cascade",
]
METHOD_SHORT = {
    "suql_baseline": "SUQL baseline",
    "suql_v1_two_level_cascade": "SUQL V1",
    "trummer_baseline_adaptive_block_join": "Trummer baseline",
    "trummer_v1_structured_two_level_cascade": "Trummer V1",
}
METHOD_COLORS = {
    "suql_baseline": "#7AA6C2",
    "suql_v1_two_level_cascade": "#315A7D",
    "trummer_baseline_adaptive_block_join": "#E8A36A",
    "trummer_v1_structured_two_level_cascade": "#A64B20",
}
QUALITY_METRICS = ["precision", "recall", "f1"]

# Amazon's aggregate.csv uses short method identifiers; IMDb uses the full ones.
METHOD_ALIASES = {
    "suql_baseline": "suql_baseline",
    "suql_v1": "suql_v1_two_level_cascade",
    "trummer_baseline": "trummer_baseline_adaptive_block_join",
    "trummer_v1": "trummer_v1_structured_two_level_cascade",
}


def load_dataset(dataset_dir: Path) -> pd.DataFrame:
    rows = []
    for pair in PAIR_ORDER:
        aggregate_path = dataset_dir / pair / "aggregate.csv"
        if not aggregate_path.exists():
            continue
        frame = pd.read_csv(aggregate_path)
        if "implementation" not in frame.columns:
            frame = frame.rename(columns={"method": "implementation"})
        frame["implementation"] = frame["implementation"].map(lambda value: METHOD_ALIASES.get(value, value))
        frame["pair"] = pair
        # Recompute uniformly from raw seconds rather than trusting whatever cost
        # columns (if any) are already in aggregate.csv: Amazon's older runs predate
        # the cost-accounting feature and have none, IMDb's have them, and this way
        # both datasets land on the exact same $/accelerator-hour basis.
        usd_per_second = ACCELERATOR_USD_PER_HOUR / 3600
        frame["total_cost_usd"] = (frame["cheap_seconds"] + frame["expensive_seconds"]) * usd_per_second
        rows.append(frame)
    combined = pd.concat(rows, ignore_index=True)
    combined["pair"] = pd.Categorical(combined["pair"], PAIR_ORDER, ordered=True)
    combined["implementation"] = pd.Categorical(combined["implementation"], METHOD_ORDER, ordered=True)
    return combined.sort_values(["pair", "implementation"])


def plot_quality_by_pair(frame: pd.DataFrame, dataset_label: str, path: Path) -> None:
    pair_labels = [PAIR_LABELS[pair] for pair in PAIR_ORDER]
    x = np.arange(len(PAIR_ORDER))
    width = 0.19
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), sharey=True)
    for ax, metric in zip(axes, QUALITY_METRICS):
        pivot = frame.pivot(index="pair", columns="implementation", values=metric).reindex(PAIR_ORDER)
        for index, method in enumerate(METHOD_ORDER):
            values = pivot[method].to_numpy(float)
            ax.bar(
                x + (index - 1.5) * width, values, width,
                label=METHOD_SHORT[method], color=METHOD_COLORS[method],
            )
        ax.set_title(metric.title())
        ax.set_xticks(x, pair_labels, fontsize=8.5)
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Score (higher is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"{dataset_label}: quality by model pair and method", y=1.1, fontsize=15)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_f1_pair_heatmap(frames: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, len(frames), figsize=(6.5 * len(frames), 5.6), sharey=True)
    if len(frames) == 1:
        axes = [axes]
    for index, (ax, (dataset_label, frame)) in enumerate(zip(axes, frames.items())):
        pivot = frame.pivot(index="pair", columns="implementation", values="f1").reindex(
            index=PAIR_ORDER, columns=METHOD_ORDER,
        )
        image = ax.imshow(pivot.to_numpy(float), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(METHOD_ORDER)), [METHOD_SHORT[method] for method in METHOD_ORDER], rotation=25, ha="right")
        if index == 0:
            ax.set_yticks(range(len(PAIR_ORDER)), [PAIR_LABELS[pair].replace("\n", " ") for pair in PAIR_ORDER], fontsize=8.5)
        for row in range(len(PAIR_ORDER)):
            for col in range(len(METHOD_ORDER)):
                value = pivot.to_numpy(float)[row, col]
                ax.text(
                    col, row, f"{value:.3f}", ha="center", va="center", fontsize=9,
                    color="white" if value > 0.55 else "#111111",
                )
        ax.set_title(dataset_label)
    fig.suptitle("F1 across model pairs and methods", fontsize=15)
    fig.subplots_adjust(wspace=0.35, right=0.88)
    colorbar_axes = fig.add_axes((0.91, 0.15, 0.018, 0.7))
    fig.colorbar(image, cax=colorbar_axes, label="F1 (higher is better)")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


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


def _cost_axis_scale(costs: np.ndarray) -> bool:
    positive = costs[costs > 0]
    return len(positive) > 0 and positive.max() / positive.min() >= 10


def plot_cost_f1_pareto(frame: pd.DataFrame, dataset_label: str, path: Path) -> None:
    """Cost-vs-F1 Pareto front across all 4 model pairs x 4 methods for one dataset."""
    costs = frame["total_cost_usd"].to_numpy(float)
    f1s = frame["f1"].to_numpy(float)
    frontier = pareto_frontier(costs, f1s)

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    for _, row in frame.reset_index(drop=True).iterrows():
        ax.scatter(
            row.total_cost_usd, row.f1, s=170, zorder=3,
            color=METHOD_COLORS[row.implementation], marker=PAIR_MARKERS[row.pair],
            edgecolor="black", linewidth=1,
        )
    if len(frontier) > 1:
        ax.plot(costs[frontier], f1s[frontier], "--", color="#333333", alpha=.6, zorder=2, label="Pareto frontier")

    use_log = _cost_axis_scale(costs)
    if use_log:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:g}"))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        xlabel = "Estimated cost per question (USD, log scale; lower is better)"
    else:
        xlabel = "Estimated cost per question (USD; lower is better)"
    ax.margins(x=0.15)
    ax.set(title=f"{dataset_label}: cost-F1 Pareto front across model pairs", xlabel=xlabel,
           ylabel="F1 (higher is better)", ylim=(-.03, 1.03))
    ax.grid(alpha=.25, which="both" if use_log else "major")

    method_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=METHOD_COLORS[m],
                   markeredgecolor="black", markersize=10, label=METHOD_SHORT[m])
        for m in METHOD_ORDER
    ]
    pair_handles = [
        plt.Line2D([0], [0], marker=PAIR_MARKERS[p], color="w", markerfacecolor="#999999",
                   markeredgecolor="black", markersize=10, label=PAIR_LABELS[p].replace("\n", " "))
        for p in PAIR_ORDER
    ]
    frontier_handles = [plt.Line2D([0], [0], linestyle="--", color="#333333", alpha=.6, label="Pareto frontier")] if len(frontier) > 1 else []
    ax.legend(handles=method_handles + pair_handles + frontier_handles, loc="upper center",
              bbox_to_anchor=(.5, -.14), ncol=3, fontsize=8.5)
    fig.text(.5, .01, f"Color = method, shape = model pair. Compute estimate uses ${ACCELERATOR_USD_PER_HOUR:.2f}/accelerator-hour.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .14, 1, 1))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_cost_f1_pareto_combined(frames: dict[str, pd.DataFrame], path: Path) -> None:
    """Both datasets' cost-F1 Pareto fronts on one shared axis.

    Averaged over the 4 model pairs per method first -- the per-dataset plots
    already show all 16 raw (pair, method) points, and overlaying 32 points
    plus two frontiers on one axis is unreadable at that granularity.
    """
    dataset_markers = {label: marker for label, marker in zip(frames, ("o", "^"))}
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    all_costs = []
    for dataset_label, frame in frames.items():
        agg = frame.groupby("implementation", as_index=False)[["total_cost_usd", "f1"]].mean()
        agg["implementation"] = pd.Categorical(agg["implementation"], METHOD_ORDER, ordered=True)
        agg = agg.sort_values("implementation")
        costs = agg["total_cost_usd"].to_numpy(float)
        f1s = agg["f1"].to_numpy(float)
        all_costs.append(costs)
        frontier = pareto_frontier(costs, f1s)
        for _, row in agg.reset_index(drop=True).iterrows():
            ax.scatter(
                row.total_cost_usd, row.f1, s=190, zorder=3,
                color=METHOD_COLORS[row.implementation], marker=dataset_markers[dataset_label],
                edgecolor="black", linewidth=1,
            )
        if len(frontier) > 1:
            ax.plot(costs[frontier], f1s[frontier], "--", color="#333333", alpha=.6, zorder=2)

    use_log = _cost_axis_scale(np.concatenate(all_costs))
    if use_log:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:g}"))
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        xlabel = "Estimated cost per question (USD, log scale; lower is better)"
    else:
        xlabel = "Estimated cost per question (USD; lower is better)"
    ax.margins(x=0.15)
    ax.set(title="Cost-F1 Pareto front, both datasets\n(mean over the 4 model pairs)", xlabel=xlabel,
           ylabel="F1 (higher is better)", ylim=(-.03, 1.03))
    ax.grid(alpha=.25, which="both" if use_log else "major")

    method_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=METHOD_COLORS[m],
                   markeredgecolor="black", markersize=10, label=METHOD_SHORT[m])
        for m in METHOD_ORDER
    ]
    dataset_handles = [
        plt.Line2D([0], [0], marker=marker, color="w", markerfacecolor="#999999",
                   markeredgecolor="black", markersize=10, label=label)
        for label, marker in dataset_markers.items()
    ]
    ax.legend(handles=method_handles + dataset_handles, loc="upper center",
              bbox_to_anchor=(.5, -.14), ncol=3, fontsize=8.5)
    fig.text(.5, .01, f"Color = method, shape = dataset. Compute estimate uses ${ACCELERATOR_USD_PER_HOUR:.2f}/accelerator-hour.",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .14, 1, 1))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    frames: dict[str, pd.DataFrame] = {}
    for dataset_key, dataset_dir in DATASETS.items():
        frame = load_dataset(dataset_dir)
        dataset_label = "IMDb" if dataset_key == "imdb" else "Amazon Fashion"
        frames[dataset_label] = frame
        plot_quality_by_pair(frame, dataset_label, HERE / f"{dataset_key}_quality_by_pair.png")
        plot_cost_f1_pareto(frame, dataset_label, HERE / f"{dataset_key}_cost_f1_pareto.png")
    plot_f1_pair_heatmap(frames, HERE / "f1_by_pair_heatmap.png")
    plot_cost_f1_pareto_combined(frames, HERE / "cost_f1_pareto_both_datasets.png")


if __name__ == "__main__":
    main()
