#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from common import (
    APPROACH_ROOT,
    benchmark,
    cpu_seconds,
    load_movies,
    load_reviews,
    question_data_dir,
    question_dir,
    write_csv,
)


def run_suql_baseline(args: argparse.Namespace, spec: dict, output_dir: Path) -> dict:
    engine_dir = APPROACH_ROOT / "project SUQL" / "baseline"
    metrics_path = output_dir / "engine_metrics.json"
    os.environ["SUQL_DATA_PATH"] = str(
        question_data_dir(args.question_dir) / "imdb_joined.csv"
    )
    os.environ["SUQL_API_BASE"] = args.api_base
    os.environ["SUQL_MODEL"] = args.expensive_model
    os.environ["SUQL_METRICS_PATH"] = str(metrics_path)
    sys.path.insert(0, str(engine_dir))
    import suql_engine

    suql_engine._answer_cache.clear()
    started_cpu = cpu_seconds()
    started_wall = time.perf_counter()
    results = suql_engine.ask_with_suql(
        spec["suql_query"],
        output_csv=str(output_dir / "found_rows.csv"),
        verbose=True,
    )
    elapsed = time.perf_counter() - started_wall
    engine_metrics = json.loads(metrics_path.read_text())
    return {
        "implementation": "suql_baseline",
        "mode": "llm",
        "model": args.expensive_model,
        "cheap_model": "",
        "expensive_model": args.expensive_model,
        "cpu_seconds": cpu_seconds() - started_cpu,
        "engine_seconds": float(engine_metrics["engine_seconds"]),
        "wall_seconds": elapsed,
        "llm_calls": int(engine_metrics["llm_prompts_issued"]),
        "block_join_calls": 0,
        "cheap_calls": 0,
        "expensive_calls": int(engine_metrics["llm_prompts_issued"]),
        "final_answer_rows": int(len(results)),
        "found_movie_ids": sorted(results["movie_id"].astype(str).unique()),
        "structured_candidates": int(engine_metrics["structured_candidates"]),
        "nonempty_fallback_rows": int(engine_metrics.get("nonempty_fallback_rows", 0)),
    }


def _suql_calibration_summary(engine_metrics: dict) -> dict:
    """Per-run calibration outcome of SUQL v1 (one semantic predicate per query).

    ``cheap_score_mode`` records how the cheap scores were obtained:
    ``native_logprobs`` means real log-odds; ``label_only`` means the server
    returned no log-probs and every score is just +-2.
    """
    usages = list((engine_metrics.get("model_usage_by_question") or {}).values())
    usage = usages[0] if usages else {}
    return {
        "calibration_mode": usage.get("calibration_mode", ""),
        "calibration_activated": bool(usage.get("calibration_activated", False)),
        "calibration_labelled_count": int(usage.get("calibration_labelled_count", 0)),
        "calibration_reused_labels": int(usage.get("calibration_reused_labels", 0)),
        "calibration_accept_precision_lower": float(usage.get("calibration_accept_precision_lower", 0.0)),
        "calibration_reject_recall_lower": float(usage.get("calibration_reject_recall_lower", 0.0)),
        "cheap_score_mode": usage.get("cheap_score_mode", ""),
    }


def run_suql_v1(args: argparse.Namespace, spec: dict, output_dir: Path) -> dict:
    engine_dir = APPROACH_ROOT / "project SUQL" / "v1"
    metrics_path = output_dir / "engine_metrics.json"
    os.environ["SUQL_DATA_PATH"] = str(
        question_data_dir(args.question_dir) / "imdb_joined.csv"
    )
    os.environ["SUQL_API_BASE"] = args.api_base
    os.environ["SUQL_MODEL"] = args.expensive_model
    os.environ["SUQL_EXPENSIVE_MODEL"] = args.expensive_model
    os.environ["SUQL_CHEAP_MODEL"] = args.cheap_model
    os.environ["SUQL_CASCADE_TARGET"] = str(args.cascade_target)
    os.environ["SUQL_CALIBRATION_BUDGET"] = str(args.calibration_budget)
    os.environ["SUQL_REQUEST_TIMEOUT"] = str(args.request_timeout)
    if args.manual_confidence_threshold is None:
        os.environ.pop("SUQL_MANUAL_CONFIDENCE_THRESHOLD", None)
    else:
        os.environ["SUQL_MANUAL_CONFIDENCE_THRESHOLD"] = str(args.manual_confidence_threshold)
    os.environ["SUQL_METRICS_PATH"] = str(metrics_path)
    sys.path.insert(0, str(engine_dir))
    import suql_engine

    suql_engine._answer_cache.clear()
    started_cpu = cpu_seconds()
    started_wall = time.perf_counter()
    results = suql_engine.ask_with_suql(
        spec["suql_query"],
        output_csv=str(output_dir / "found_rows.csv"),
        verbose=True,
    )
    elapsed = time.perf_counter() - started_wall
    engine_metrics = json.loads(metrics_path.read_text())
    cheap_calls = int(engine_metrics.get("cheap_score_calls", 0))
    expensive_calls = int(engine_metrics.get("llm_full_calls", 0))
    return {
        "implementation": "suql_v1_two_level_cascade",
        "mode": "llm",
        "model": f"{args.cheap_model}->{args.expensive_model}",
        "cheap_model": args.cheap_model,
        "expensive_model": args.expensive_model,
        "cpu_seconds": cpu_seconds() - started_cpu,
        "engine_seconds": float(engine_metrics["engine_seconds"]),
        "wall_seconds": elapsed,
        "llm_calls": cheap_calls + expensive_calls,
        "block_join_calls": 0,
        "cheap_calls": cheap_calls,
        "expensive_calls": expensive_calls,
        "cheap_seconds": float(engine_metrics.get("cheap_seconds", 0.0)),
        "expensive_seconds": float(engine_metrics.get("expensive_seconds", 0.0)),
        "cheap_early_accepts": int(engine_metrics.get("cheap_early_accept", 0)),
        "cheap_early_rejects": int(engine_metrics.get("cheap_early_reject", 0)),
        "calibration_candidates": int(engine_metrics.get("calibration_candidates", 0)),
        "calibration_expensive_calls": int(engine_metrics.get("calibration_expensive_calls", 0)),
        "calibration_expensive_accepts": int(engine_metrics.get("calibration_expensive_accepts", 0)),
        "calibration_agreement": float(engine_metrics.get("calibration_agreement", 0.0)),
        **_suql_calibration_summary(engine_metrics),
        "final_answer_rows": int(len(results)),
        "found_movie_ids": sorted(results["movie_id"].astype(str).unique()),
        "structured_candidates": int(engine_metrics["structured_candidates"]),
        "nonempty_fallback_rows": int(engine_metrics.get("nonempty_fallback_rows", 0)),
    }


def run_trummer_baseline(args: argparse.Namespace, spec: dict, output_dir: Path) -> dict:
    """Run the paper-style adaptive block join without structured pruning."""
    input_movies = load_movies(args.question_dir)
    input_reviews = load_reviews(args.question_dir)
    movies = pd.DataFrame(input_movies)
    reviews = pd.DataFrame(input_reviews)

    for name in list(sys.modules):
        if name == "trummer_join" or name.startswith("trummer_join."):
            del sys.modules[name]
    v1_root = APPROACH_ROOT / "project Trummer" / "baseline"
    sys.path.insert(0, str(v1_root))
    from trummer_join.client import ChatClient
    from trummer_join.operators import single_pass_join

    # The complete benchmark question stays inside the semantic block-join
    # predicate. No year, genre, runtime, title, or director filter is applied
    # before prompting.
    predicate = spec["question"]
    started_cpu = cpu_seconds()
    started_wall = time.perf_counter()
    # One expensive call per batch of rows: ceil(n / block_size).  The previous
    # adaptive_join issued rounds x (row-blocks x text-blocks) calls instead.
    stats, joined = single_pass_join(
        ChatClient(api_base=args.api_base, timeout=args.request_timeout),
        movies,
        reviews,
        predicate,
        args.expensive_model,
        block_size=args.max_movie_block_size,
        token_threshold=args.token_threshold,
        max_completion_tokens=args.max_completion_tokens,
    )
    elapsed = time.perf_counter() - started_wall
    joined_rows = joined.to_dict("records") if not joined.empty else []
    final_rows = []
    seen = set()
    for row in joined_rows:
        movie_id = str(row.get("movie_id", ""))
        if movie_id and movie_id not in seen:
            seen.add(movie_id)
            final_rows.append(
                {
                    key: row.get(key, "")
                    for key in ("movie_id", "title", "year", "runtime", "director", "genres")
                }
            )
    stats.to_csv(output_dir / "join_stats.csv", index=False)
    write_csv(output_dir / "joined_evidence.csv", joined_rows)
    write_csv(output_dir / "found_rows.csv", final_rows)
    return {
        "implementation": "trummer_baseline_adaptive_block_join",
        "mode": "llm",
        "operator": "adaptive_block_join",
        "model": args.expensive_model,
        "cheap_model": "",
        "expensive_model": args.expensive_model,
        "cpu_seconds": cpu_seconds() - started_cpu,
        "engine_seconds": elapsed,
        "wall_seconds": elapsed,
        "llm_calls": int(len(stats)),
        "block_join_calls": int(len(stats)),
        "cheap_calls": 0,
        "expensive_calls": int(len(stats)),
        "final_answer_rows": len(final_rows),
        "found_movie_ids": sorted(seen),
        "original_movies": len(input_movies),
        "original_reviews": len(input_reviews),
        "pruned_movies": len(input_movies),
        "pruned_reviews": len(input_reviews),
        "structured_pruning": False,
        "semantic_condition_location": "paper_style_block_join_prompt",
        "prompt_tokens": int(stats["tokens_read"].sum()) if "tokens_read" in stats else 0,
        "completion_tokens": int(stats["tokens_written"].sum()) if "tokens_written" in stats else 0,
        "nonempty_fallback_rows": int(
            any(str(row.get("match_source", "")) == "nonempty_fallback" for row in joined_rows)
        ),
    }


def run_trummer_v1(args: argparse.Namespace, spec: dict, output_dir: Path) -> dict:
    for name in list(sys.modules):
        if name == "trummer_join" or name.startswith("trummer_join."):
            del sys.modules[name]
    v1_root = APPROACH_ROOT / "project Trummer" / "v1"
    sys.path.insert(0, str(v1_root))
    from trummer_join.cascade import CascadeConfig, CascadeJoin, metrics_dict
    from trummer_join.structured_filter import prune_movie_frame

    input_movies = load_movies(args.question_dir)
    input_reviews = load_reviews(args.question_dir)
    input_movies_frame = pd.DataFrame(input_movies)
    input_reviews_frame = pd.DataFrame(input_reviews)
    movies_frame, pruning = prune_movie_frame(
        input_movies_frame,
        spec["question"],
        api_base=args.api_base,
        parser_model=args.structured_parser_model or args.cheap_model,
        request_timeout=args.request_timeout,
        use_llm=not args.disable_llm_structured_parser,
    )
    movie_ids = set(movies_frame["movie_id"].astype(str))
    reviews_frame = pd.DataFrame(
        [
            row
            for row in input_reviews
            if str(row.get("tconst", "")) in movie_ids
        ],
        columns=input_reviews_frame.columns,
    ).reset_index(drop=True)
    movies = movies_frame.to_dict("records")
    reviews = reviews_frame.to_dict("records")
    config = CascadeConfig(
        api_base=args.api_base,
        cheap_model=args.cheap_model,
        expensive_model=args.expensive_model,
        cascade_target=args.cascade_target,
        calibration_budget=args.calibration_budget,
        manual_confidence_threshold=args.manual_confidence_threshold,
        cheap_batch_size=args.cheap_batch_size,
        expensive_batch_size=args.expensive_batch_size,
        request_timeout=args.request_timeout,
    )
    started_cpu = cpu_seconds()
    started_wall = time.perf_counter()
    rows, decisions, metrics = CascadeJoin(config).run(
        movies,
        reviews,
        spec["semantic_question"],
    )
    elapsed = time.perf_counter() - started_wall
    final_rows = []
    seen = set()
    for row in rows:
        movie_id = str(row.get("movie_id", ""))
        if movie_id in seen:
            continue
        seen.add(movie_id)
        final_rows.append(
            {
                key: row.get(key, "")
                for key in (
                    "movie_id",
                    "title",
                    "year",
                    "runtime",
                    "director",
                    "genres",
                    "match_source",
                )
            }
        )
    write_csv(
        output_dir / "cascade_decisions.csv",
        [asdict(item) for item in decisions],
    )
    write_csv(output_dir / "joined_evidence.csv", rows)
    write_csv(output_dir / "found_rows.csv", final_rows)
    return {
        "implementation": "trummer_v1_structured_two_level_cascade",
        "mode": "llm",
        "model": f"{args.cheap_model}->{args.expensive_model}",
        "cheap_model": args.cheap_model,
        "expensive_model": args.expensive_model,
        **metrics_dict(metrics),
        "cpu_seconds": cpu_seconds() - started_cpu,
        "engine_seconds": elapsed,
        "wall_seconds": elapsed,
        "llm_calls": metrics.cheap_calls + metrics.expensive_calls,
        "block_join_calls": metrics.expensive_calls,
        "final_answer_rows": len(final_rows),
        "found_movie_ids": sorted(seen),
        "original_movies": len(input_movies),
        "original_reviews": len(input_reviews),
        "pruned_movies": len(movies),
        "pruned_reviews": len(reviews),
        "structured_filters": [item.as_dict() for item in pruning.filters],
        "structured_pruning": pruning.as_dict(),
        "structured_condition_location": "deterministic_prefilter",
        "join_condition_location": "deterministic_prefilter",
        "semantic_condition_location": "batch_cascade",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        required=True,
        choices=["suql_baseline", "suql_v1", "trummer_baseline", "trummer_v1"],
    )
    parser.add_argument("--question-dir", required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:11434")
    parser.add_argument("--cheap-model", default="ollama/gemma4:e2b")
    parser.add_argument("--expensive-model", default="ollama/gemma4:e4b")
    parser.add_argument("--structured-parser-model")
    parser.add_argument("--disable-llm-structured-parser", action="store_true")
    parser.add_argument("--cascade-target", type=float, default=0.8)
    parser.add_argument("--calibration-budget", type=int, default=20)
    parser.add_argument("--manual-confidence-threshold", type=float)
    parser.add_argument("--cheap-accept-threshold", type=float, default=3.0)
    parser.add_argument("--cheap-reject-threshold", type=float, default=-1.5)
    parser.add_argument("--cheap-batch-size", type=int, default=8)
    parser.add_argument("--expensive-batch-size", type=int, default=32)
    parser.add_argument("--max-expensive-calls", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--token-threshold", type=int, default=4096)
    parser.add_argument("--max-completion-tokens", type=int, default=256)
    parser.add_argument("--max-movie-block-size", type=int, default=25)
    parser.add_argument("--max-review-block-size", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = benchmark(args.question_dir)
    if args.method == "suql_baseline":
        payload = run_suql_baseline(args, spec, output_dir)
    elif args.method == "suql_v1":
        payload = run_suql_v1(args, spec, output_dir)
    elif args.method == "trummer_baseline":
        payload = run_trummer_baseline(args, spec, output_dir)
    else:
        payload = run_trummer_v1(args, spec, output_dir)
    payload["question_dir"] = args.question_dir
    payload["question"] = spec["question"]
    (output_dir / "run_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
