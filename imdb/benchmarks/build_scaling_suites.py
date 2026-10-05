#!/usr/bin/env python3
"""Build constant-prevalence scaled-up variants of the 10q questions.

``scale_x<S>`` holds, for each selected question, a pool S times larger than the
10q pool with the same composition: 12*S true positives (structured AND semantic),
28*S structured negatives (so 40*S rows survive the structured filter, 30 percent
positives, like 10q), and 30*S + 30*S distractors that the structured filter drops.
Labels come from the same declared deterministic lexical rules as 10q
(build_catalog.py), so ground truth exists for every added row.

Rows are unique within a question; different questions may share rows.

Why a subset of questions and S <= 4: the corpus has only 3,209 distinct movies, and
a question needs 12*S distinct movies satisfying structured AND semantic conditions
and 28*S distinct structured negatives (mystery has 12 and 234, so it cannot scale
at all). The default questions (2, 4, 7, 8, 10) all admit S = 4, i.e. 160 candidates
(q1, q3, q6, q9 stop at S = 3, q5 cannot scale).

Usage: build_scaling_suites.py [--scales 1,2,4] [--questions 2,4,7,8,10]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import build_catalog as base

BENCH = Path(__file__).resolve().parent
DATA_ROOT = BENCH.parent / "data" / "subdatasets"
CLASS_SIZES = {"pos": 12, "struct_neg": 28, "sem_only": 30, "double_neg": 30}


def build_question(spec: base.QuestionSpec, source: pd.DataFrame, used: set[str], scale: int,
                   suite: str, qid: str) -> dict:
    structured = base.structured_mask(source, spec)
    semantic = base.semantic_mask(source, spec)
    selected = pd.concat([
        base.take_unique(source[structured & semantic], CLASS_SIZES["pos"] * scale, used),
        base.take_unique(source[structured & ~semantic], CLASS_SIZES["struct_neg"] * scale, used),
        base.take_unique(source[~structured & semantic], CLASS_SIZES["sem_only"] * scale, used),
        base.take_unique(source[~structured & ~semantic], CLASS_SIZES["double_neg"] * scale, used),
    ], ignore_index=True)
    selected["structured_match"] = base.structured_mask(selected, spec)
    selected["semantic_label"] = base.semantic_mask(selected, spec)
    selected["ground_truth"] = (selected["structured_match"] & selected["semantic_label"]).astype(int)
    selected = selected.sort_values(["structured_match", "movie_id"], ascending=[False, True])

    data_dir = DATA_ROOT / suite / qid
    data_dir.mkdir(parents=True, exist_ok=True)
    selected[base.OUTPUT_COLUMNS].to_csv(data_dir / "imdb_joined.csv", index=False)
    selected[["movie_id", "title", "director", "year", "runtime", "genres"]].to_csv(
        data_dir / "imdb_structured_joined.csv", index=False)
    selected[["movie_id", "review"]].rename(columns={"movie_id": "tconst"}).to_csv(
        data_dir / "imdb_reviews.csv", index=False)
    annotations = selected[["movie_id", "source_row", "title", "year", "runtime", "genres",
                            "structured_match", "semantic_label", "ground_truth"]].copy()
    annotations["annotation_source"] = "deterministic case-insensitive lexical policy over IMDb review text"
    annotations.to_csv(data_dir / "annotations.csv", index=False)
    truth = annotations[annotations.ground_truth.eq(1)]
    truth[["movie_id", "title", "ground_truth"]].to_csv(data_dir / "ground_truth.csv", index=False)

    benchmark = {
        "benchmark_id": f"{suite}_{qid}", "difficulty": spec.difficulty,
        "difficulty_label": spec.difficulty_label, "question": spec.question,
        "semantic_question": spec.semantic_question, "suql_query": spec.suql_query,
        "structured_filters": list(spec.structured_filters), "semantic_task": spec.semantic_task,
        "semantic_annotation_patterns": list(spec.semantic_patterns),
        "annotation_policy": "Deterministic lexical annotation; no evaluated LLM used",
        "row_count": int(len(selected)), "structured_candidate_count": int(selected.structured_match.sum()),
        "semantic_positive_count": int(selected.semantic_label.sum()),
        "ground_truth_movie_ids": sorted(truth.movie_id.astype(str)),
    }
    question_dir = BENCH / suite / "per_question" / qid
    question_dir.mkdir(parents=True, exist_ok=True)
    (question_dir / "benchmark.json").write_text(json.dumps(benchmark, indent=2) + "\n")
    return benchmark


def build_suite(scale: int, numbers: list[int], source: pd.DataFrame) -> None:
    suite = f"scale_x{scale}"
    specs = [spec for spec in base.QUESTIONS if spec.number in numbers]
    built: dict[int, dict] = {}
    ids = {spec.number: f"q_{index:02d}" for index, spec in enumerate(sorted(specs, key=lambda s: s.number), 1)}
    for spec in specs:
        # Rows are unique within a question but may repeat across questions: 10q's
        # cross-question disjointness would exhaust the overlapping genre pools
        # (Animation/Family, Drama/Drama) at larger scales.
        built[spec.number] = build_question(spec, source, set(), scale, suite, ids[spec.number])
    manifest = {
        "suite": suite, "question_count": len(specs), "scale": scale,
        "source_questions": sorted(numbers),
        "questions": [
            {"id": ids[s.number], "directory": ids[s.number], "benchmark_id": built[s.number]["benchmark_id"],
             "difficulty": s.difficulty, "difficulty_label": s.difficulty_label, "question": s.question,
             "semantic_task": s.semantic_task, "structured_filters": list(s.structured_filters),
             "row_count": built[s.number]["row_count"],
             "structured_candidate_count": built[s.number]["structured_candidate_count"],
             "ground_truth_count": len(built[s.number]["ground_truth_movie_ids"])}
            for s in sorted(specs, key=lambda s: s.number)
        ],
        "dataset_policy": f"10q composition scaled by {scale}: {CLASS_SIZES['pos'] * scale} positives, "
                          f"{CLASS_SIZES['struct_neg'] * scale} structured negatives and "
                          f"{(CLASS_SIZES['sem_only'] + CLASS_SIZES['double_neg']) * scale} distractors per question.",
    }
    (BENCH / suite / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{suite}: {len(specs)} questions, {40 * scale} candidates / {12 * scale} positives each")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", default="1,2,4")
    parser.add_argument("--questions", default="2,4,7,8,10")
    args = parser.parse_args()
    numbers = [int(x) for x in args.questions.split(",")]
    source = base.load_source()
    for scale in [int(x) for x in args.scales.split(",")]:
        build_suite(scale, numbers, source)


if __name__ == "__main__":
    main()
