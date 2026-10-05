#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from common import SUITE_ROOT, question_dir
from repetitions import run_repeated


METHOD_DIRS = {
    "suql_baseline": "suql_baseline",
    "suql_v1": "suql_v1_two_level_cascade",
    "trummer_baseline": "trummer_baseline",
    "trummer_v1": "trummer_v1_structured_two_level_cascade",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--api-base", default="http://127.0.0.1:11434")
    parser.add_argument("--cheap-model", default="ollama/gemma4:e2b")
    parser.add_argument("--expensive-model", default="ollama/gemma4:26b")
    parser.add_argument("--structured-parser-model")
    parser.add_argument("--disable-llm-structured-parser", action="store_true")
    parser.add_argument("--cascade-target", type=float, default=0.8)
    parser.add_argument("--calibration-budget", type=int, default=20)
    parser.add_argument("--manual-confidence-threshold", type=float)
    parser.add_argument("--cheap-batch-size", type=int, default=8)
    parser.add_argument("--expensive-batch-size", type=int, default=32)
    parser.add_argument("--max-expensive-calls", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=3600)
    parser.add_argument("--token-threshold", type=int, default=4096)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--max-movie-block-size", type=int, default=25)
    parser.add_argument("--max-review-block-size", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--methods", nargs="+", choices=sorted(METHOD_DIRS), default=list(METHOD_DIRS))
    parser.add_argument(
        "--keep-run-artifacts",
        action="store_true",
        help="Keep raw predictions, logs, and metric JSON files after plotting.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    manifest = json.loads((SUITE_ROOT / "manifest.json").read_text())
    questions = [str(item["directory"]) for item in manifest["questions"]]
    if len(questions) != int(manifest["question_count"]):
        raise RuntimeError("Manifest question_count does not match its question list")
    output_dir = Path(args.output_dir).resolve()
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", str(SUITE_ROOT / ".mplconfig"))
    structured_parser_model = args.structured_parser_model or args.cheap_model

    for question in questions:
        spec = json.loads((question_dir(question) / "benchmark.json").read_text())
        truth = set(spec["ground_truth_movie_ids"])
        if len(truth) < 10:
            raise RuntimeError(f"{question} has fewer than 10 ground-truth movies")
        for method in args.methods:
            command = [
                args.python, str(Path(__file__).with_name("run_method.py")),
                "--method", method, "--question-dir", question,
                "--api-base", args.api_base,
                "--cheap-model", args.cheap_model,
                "--expensive-model", args.expensive_model,
                "--structured-parser-model", structured_parser_model,
                "--cascade-target", str(args.cascade_target),
                "--calibration-budget", str(args.calibration_budget),
                "--cheap-batch-size", str(args.cheap_batch_size),
                "--expensive-batch-size", str(args.expensive_batch_size),
                "--max-expensive-calls", str(args.max_expensive_calls),
                "--request-timeout", str(args.request_timeout),
                "--token-threshold", str(args.token_threshold),
                "--max-completion-tokens", str(args.max_completion_tokens),
                "--max-movie-block-size", str(args.max_movie_block_size),
                "--max-review-block-size", str(args.max_review_block_size),
                "--output-dir", str(output_dir / "per_question" / question / METHOD_DIRS[method]),
            ]
            if args.disable_llm_structured_parser:
                command.append("--disable-llm-structured-parser")
            if args.manual_confidence_threshold is not None:
                command.extend(["--manual-confidence-threshold", str(args.manual_confidence_threshold)])
            run_repeated(command, env, SUITE_ROOT, args.repetitions, truth)

    evaluate_command = [
        args.python, str(Path(__file__).with_name("evaluate_and_plot.py")),
        "--suite-root", str(SUITE_ROOT), "--outputs-dir", str(output_dir),
    ]
    if args.keep_run_artifacts:
        evaluate_command.append("--keep-run-artifacts")
    subprocess.run(evaluate_command, env=env, check=True)


if __name__ == "__main__":
    main()
