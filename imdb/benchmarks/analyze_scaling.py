#!/usr/bin/env python3
"""Summarise the scaling study: does the cascade's fixed overhead ever pay off?

Reads ``<suite>/outputs/<suite>_<tag>_<reps>rep/comparison.csv`` for every scale and
model-pair tag, and reports, per pair and pool size, each cascade relative to its own
baseline (wall time, LLM calls, F1). A cascade "pays off" where its relative wall time
is below 1.0 at no F1 loss.

Usage: analyze_scaling.py --tags gemma,llama [--reps 2] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

BENCH = Path(__file__).resolve().parent
IMPL = {
    "suql_baseline": "SUQL",
    "suql_v1_two_level_cascade": "CasSuql",
    "trummer_baseline_adaptive_block_join": "Trummer",
    "trummer_v1_structured_two_level_cascade": "CasTrummer",
}
PAIRS = [("CasSuql", "SUQL"), ("CasTrummer", "Trummer")]


def load(path: Path) -> dict[str, dict[str, float]]:
    rows = list(csv.DictReader(path.open()))
    out: dict[str, dict[str, float]] = {}
    for method in IMPL.values():
        picked = [r for r in rows if IMPL.get(r["implementation"]) == method]
        if not picked:
            continue
        out[method] = {
            key: statistics.mean(float(r[key]) for r in picked)
            for key in ("f1", "precision", "recall", "wall_seconds", "llm_calls", "expensive_calls", "cheap_calls")
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tags", required=True, help="comma list of pair tags used in --output-name")
    parser.add_argument("--scales", default="1,2,4")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--out-dir", default=str(BENCH / "scaling_results"))
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scales = [int(s) for s in args.scales.split(",")]

    lines = ["| pair | candidates | method | F1 | F1 vs baseline | wall s | wall vs baseline | calls vs baseline | expensive calls vs baseline |",
             "|---|---|---|---|---|---|---|---|---|"]
    series: dict[tuple[str, str], list[tuple[int, float, float]]] = {}
    for tag in args.tags.split(","):
        for scale in scales:
            suite = f"scale_x{scale}"
            path = BENCH / suite / "outputs" / f"{suite}_{tag}_{args.reps}rep" / "comparison.csv"
            if not path.exists():
                print(f"missing: {path}")
                continue
            data = load(path)
            for cascade, base in PAIRS:
                if cascade not in data or base not in data:
                    continue
                c, b = data[cascade], data[base]
                lines.append(
                    f"| {tag} | {40 * scale} | {cascade} | {c['f1']:.3f} | {c['f1'] - b['f1']:+.3f} | "
                    f"{c['wall_seconds']:.1f} | {c['wall_seconds'] / b['wall_seconds']:.2f} | "
                    f"{c['llm_calls'] / b['llm_calls']:.2f} | {c['expensive_calls'] / b['expensive_calls']:.2f} |"
                )
                series.setdefault((tag, cascade), []).append(
                    (40 * scale, c["wall_seconds"] / b["wall_seconds"], c["f1"] - b["f1"])
                )
    table = "\n".join(lines)
    (out_dir / "scaling_table.md").write_text(table + "\n")
    print(table)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for (tag, cascade), points in sorted(series.items()):
        points.sort()
        xs = [p[0] for p in points]
        axes[0].plot(xs, [p[1] for p in points], marker="o", label=f"{cascade} ({tag})")
        axes[1].plot(xs, [p[2] for p in points], marker="o", label=f"{cascade} ({tag})")
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set(xlabel="candidates per question", ylabel="wall time / own baseline", xscale="log",
                title="Below 1.0: the cascade is faster")
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1].set(xlabel="candidates per question", ylabel="F1 minus own baseline", xscale="log",
                title="Quality cost of the cascade")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "scaling.png", dpi=160)
    print(f"wrote {out_dir / 'scaling_table.md'} and scaling.png")


if __name__ == "__main__":
    main()
