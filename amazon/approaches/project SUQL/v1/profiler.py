from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scorer import OllamaLogOddsScorer, parse_label


DEFAULT_CHEAP_MODEL = os.environ.get("SUQL_CHEAP_MODEL", "ollama/gemma2:2b")
DEFAULT_EXPENSIVE_MODEL = os.environ.get(
    "SUQL_EXPENSIVE_MODEL",
    os.environ.get("SUQL_MODEL", "ollama/phi4-mini"),
)
DEFAULT_API_BASE = os.environ.get("SUQL_API_BASE", "http://localhost:11434")


@dataclasses.dataclass(frozen=True)
class CascadeThreshold:
    question: str
    cheap_accept_threshold: float
    cheap_reject_threshold: float
    accept_precision_lower: float
    reject_precision_lower: float
    accept_precision_target: float
    reject_precision_target: float
    credible_level: float
    labelled_examples: int
    early_accept_count: int
    early_reject_count: int
    expensive_count: int

    def to_json_dict(self) -> dict:
        return dataclasses.asdict(self)


def beta_lower_bound(successes: int, failures: int, credible_level: float) -> float:
    from scipy.stats import beta

    return float(beta.ppf(1.0 - credible_level, 1 + successes, 1 + failures))


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    finite = scores[np.isfinite(scores)]
    if len(finite) == 0:
        raise ValueError("No finite scores available.")
    return np.unique(np.concatenate([finite, np.linspace(float(finite.min()), float(finite.max()), 200)]))


def _fit_accept_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    precision_target: float,
    credible_level: float,
) -> tuple[float, float, int]:
    best: tuple[float, float, int] | None = None
    for threshold in _candidate_thresholds(scores):
        selected = scores >= threshold
        selected_count = int(selected.sum())
        if selected_count == 0:
            continue
        tp = int(np.logical_and(selected, labels == 1).sum())
        fp = selected_count - tp
        lower = beta_lower_bound(tp, fp, credible_level)
        if lower < precision_target:
            continue
        candidate = (float(threshold), lower, selected_count)
        if best is None or candidate[2] > best[2] or (candidate[2] == best[2] and candidate[0] < best[0]):
            best = candidate
    if best is None:
        return math.inf, 1.0, 0
    return best


def _fit_reject_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    precision_target: float,
    credible_level: float,
) -> tuple[float, float, int]:
    best: tuple[float, float, int] | None = None
    for threshold in _candidate_thresholds(scores):
        selected = scores <= threshold
        selected_count = int(selected.sum())
        if selected_count == 0:
            continue
        tn = int(np.logical_and(selected, labels == 0).sum())
        fn = selected_count - tn
        lower = beta_lower_bound(tn, fn, credible_level)
        if lower < precision_target:
            continue
        candidate = (float(threshold), lower, selected_count)
        if best is None or candidate[2] > best[2] or (candidate[2] == best[2] and candidate[0] > best[0]):
            best = candidate
    if best is None:
        return -math.inf, 1.0, 0
    return best


def fit_cascade_threshold(
    question: str,
    scores: np.ndarray,
    labels: np.ndarray,
    accept_precision_target: float = 0.9,
    reject_precision_target: float = 0.9,
    credible_level: float = 0.9,
) -> CascadeThreshold:
    accept_threshold, accept_lower, early_accept_count = _fit_accept_threshold(
        scores,
        labels,
        precision_target=accept_precision_target,
        credible_level=credible_level,
    )
    reject_threshold, reject_lower, early_reject_count = _fit_reject_threshold(
        scores,
        labels,
        precision_target=reject_precision_target,
        credible_level=credible_level,
    )
    if reject_threshold >= accept_threshold:
        # Keep a real unsure band if independently fitted thresholds cross.
        midpoint = float(np.median(scores))
        reject_threshold = min(reject_threshold, midpoint)
        accept_threshold = max(accept_threshold, midpoint)
        early_accept_count = int((scores >= accept_threshold).sum())
        early_reject_count = int((scores <= reject_threshold).sum())

    expensive_count = int(len(scores) - early_accept_count - early_reject_count)
    return CascadeThreshold(
        question=question,
        cheap_accept_threshold=accept_threshold,
        cheap_reject_threshold=reject_threshold,
        accept_precision_lower=accept_lower,
        reject_precision_lower=reject_lower,
        accept_precision_target=accept_precision_target,
        reject_precision_target=reject_precision_target,
        credible_level=credible_level,
        labelled_examples=len(scores),
        early_accept_count=early_accept_count,
        early_reject_count=early_reject_count,
        expensive_count=max(expensive_count, 0),
    )


def profile_dataframe(
    df: pd.DataFrame,
    scorer: OllamaLogOddsScorer,
    accept_precision_target: float = 0.9,
    reject_precision_target: float = 0.9,
    credible_level: float = 0.9,
) -> dict:
    required = {"review", "question", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    thresholds = []
    scored_examples = []
    for question, group in df.groupby("question", sort=False):
        scores = []
        labels = []
        for row in group.itertuples(index=False):
            score = scorer.score(str(row.review), str(question))
            label = parse_label(row.label)
            scores.append(score)
            labels.append(label)
            scored_examples.append({"question": str(question), "label": label, "cheap_score": score})

        scores_arr = np.asarray(scores, dtype=float)
        labels_arr = np.asarray(labels, dtype=int)
        thresholds.append(
            fit_cascade_threshold(
                question=str(question),
                scores=scores_arr,
                labels=labels_arr,
                accept_precision_target=accept_precision_target,
                reject_precision_target=reject_precision_target,
                credible_level=credible_level,
            ).to_json_dict()
        )

    return {
        "cheap_model": scorer.model,
        "expensive_model": DEFAULT_EXPENSIVE_MODEL,
        "api_base": scorer.api_base,
        "thresholds": thresholds,
        "scored_examples": scored_examples,
    }


def save_thresholds(profile: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(profile, indent=2), encoding="utf-8")
