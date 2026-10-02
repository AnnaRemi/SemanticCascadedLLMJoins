from __future__ import annotations

import dataclasses
import json
import math
import time
import urllib.error
import urllib.request
from typing import Iterable

import pandas as pd

from .semantic_dict_context import semantic_guideline
from .threshold_fit import fit_reject_threshold

# Per-row logprob-based confidence scorer, ported from SUQL v1's
# OllamaLogOddsScorer (project SUQL/v1/scorer.py) to stay stdlib-only (see
# threshold_fit.py's header for the same rationale). Keep the two in sync if
# the underlying scoring math changes.

CHEAP_EVIDENCE_INSTRUCTIONS = (
    "Act as a high-recall first-pass semantic filter. Decide whether this review "
    "contains review-specific evidence for a Yes answer.\n"
    "Answer YES for direct evidence, synonyms, described examples, or reasonable "
    "implications. Give borderline but condition-specific evidence the benefit of "
    "the doubt. Answer NO only when review-specific support is absent, generic, "
    "based only on genre/topic, or never discusses the predicate itself. Lack of "
    "contradiction alone is not evidence. Count explicit mentions even when negated, "
    "qualified, quoted, or critical: this is evidence retrieval, not sentiment scoring.\n"
    "Return exactly one token: 1 for Yes, 0 for No."
)


def _label(token: str) -> str | None:
    value = token.strip().strip('"').strip("'").lower().rstrip(".")
    first = value.split()[0].rstrip(".,:;") if value.split() else value
    if value in {"1", "yes", "true"} or first in {"1", "yes", "true"}:
        return "1"
    if value in {"0", "no", "false"} or first in {"0", "no", "false"}:
        return "0"
    return None


def _single_score(label: str, logprob: float | None) -> float:
    if logprob is None:
        magnitude = 2.0
    else:
        probability = min(max(math.exp(float(logprob)), 0.500001), 0.999999)
        magnitude = math.log(probability) - math.log1p(-probability)
    return magnitude if label == "1" else -magnitude


def _mapping_score(mapping: dict) -> float | None:
    values: dict[str, float] = {}
    for token, logprob in mapping.items():
        label = _label(str(token))
        if label is not None:
            values[label] = float(logprob)
    if "1" in values and "0" in values:
        return values["1"] - values["0"]
    for token, logprob in mapping.items():
        label = _label(str(token))
        if label is not None:
            return _single_score(label, float(logprob))
    return None


def _list_score(items: Iterable[dict]) -> float | None:
    values: dict[str, float] = {}
    materialized = list(items)
    for item in materialized:
        label = _label(str(item.get("token", "")))
        if label is not None and "logprob" in item:
            values[label] = float(item["logprob"])
    if "1" in values and "0" in values:
        return values["1"] - values["0"]
    for item in materialized:
        label = _label(str(item.get("token", "")))
        if label is not None:
            value = item.get("logprob")
            return _single_score(label, float(value) if value is not None else None)
    return None


def extract_binary_log_odds(payload: dict) -> float:
    candidates: list[object] = []
    for choice in payload.get("choices", []):
        logprobs = choice.get("logprobs") or {}
        candidates.append(logprobs)
        if isinstance(logprobs, dict):
            candidates.extend(logprobs.get("top_logprobs") or [])
            for item in logprobs.get("content") or []:
                if isinstance(item, dict):
                    candidates.append(item.get("top_logprobs"))
    candidates.extend([payload.get("logprobs"), payload.get("response_logprobs")])
    if isinstance(payload.get("logprobs"), list):
        candidates.extend(payload["logprobs"])
    for candidate in candidates:
        if isinstance(candidate, dict):
            score = _mapping_score(candidate)
            if score is not None:
                return score
            for item in candidate.get("top_logprobs") or []:
                if isinstance(item, dict):
                    score = _mapping_score(item)
                    if score is not None:
                        return score
        elif isinstance(candidate, list):
            score = _list_score(item for item in candidate if isinstance(item, dict))
            if score is not None:
                return score
    response_label = _label(str(payload.get("response", "")))
    if response_label is not None:
        return _single_score(response_label, None)
    for choice in payload.get("choices", []):
        text = choice.get("text")
        if text is None and isinstance(choice.get("message"), dict):
            text = choice["message"].get("content")
        choice_label = _label(str(text or ""))
        if choice_label is not None:
            return _single_score(choice_label, None)
    raise ValueError("Could not extract a binary score from the model response.")


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class OllamaRowLogOddsScorer:
    """Per-row logprob confidence scorer (one candidate per request)."""

    def __init__(self, model: str, api_base: str, timeout: float = 120.0) -> None:
        self.ollama_model = model.removeprefix("ollama/")
        self.api_base = api_base.rstrip("/")
        self.timeout = timeout
        self._openai_completions_available: bool | None = None

    def score(self, review: str, predicate: str) -> float:
        prompt = self._prompt(review, predicate)
        openai_error: Exception | None = None
        if self._openai_completions_available is not False:
            try:
                payload = _post_json(
                    f"{self.api_base}/v1/completions",
                    {
                        "model": self.ollama_model,
                        "prompt": prompt,
                        "max_tokens": 1,
                        "temperature": 0,
                        "logprobs": 10,
                        "think": False,
                    },
                    self.timeout,
                )
                self._openai_completions_available = True
                return extract_binary_log_odds(payload)
            except urllib.error.HTTPError as exc:
                self._openai_completions_available = False
                if exc.code != 404:
                    openai_error = exc
            except Exception as exc:
                openai_error = exc
                self._openai_completions_available = False
        try:
            payload = _post_json(
                f"{self.api_base}/api/generate",
                {
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "think": False,
                    "options": {"temperature": 0, "num_predict": 1},
                },
                self.timeout,
            )
            return extract_binary_log_odds(payload)
        except Exception as native_error:
            if openai_error is not None:
                raise RuntimeError(
                    f"Both scoring APIs failed: {openai_error}; {native_error}"
                ) from native_error
            raise

    @staticmethod
    def _prompt(review: str, predicate: str) -> str:
        guidance = semantic_guideline(predicate)
        guidance_block = f"{guidance}\n\n" if guidance else ""
        return (
            CHEAP_EVIDENCE_INSTRUCTIONS
            + "\n\n"
            + guidance_block
            + f"Predicate: {predicate}\nReview: {review[:1800]}\n\nAnswer:"
        )


@dataclasses.dataclass(frozen=True)
class CheapFilterConfig:
    api_base: str = "http://127.0.0.1:11434"
    cheap_model: str = "gemma4:e2b"
    cascade_target: float = 0.8
    credible_level: float = 0.85
    ground_truth_ids: frozenset[str] | None = None
    manual_reject_threshold: float | None = None
    request_timeout: float = 600.0
    max_review_chars: int = 3500

    def validate(self) -> None:
        if not 0.0 < self.cascade_target <= 1.0:
            raise ValueError("cascade_target must be in (0, 1]")
        if not 0.0 < self.credible_level < 1.0:
            raise ValueError("credible_level must be in (0, 1)")


@dataclasses.dataclass
class CheapFilterMetrics:
    rows_scored: int = 0
    cheap_calls: int = 0
    cheap_seconds: float = 0.0
    cheap_failures: int = 0
    rows_rejected: int = 0
    rows_survived: int = 0
    reject_threshold: float | None = None
    reject_precision_lower: float = 0.0
    calibration_mode: str = ""
    calibration_activated: bool = False


def filter_products(
    products_frame: pd.DataFrame,
    reviews_frame: pd.DataFrame,
    predicate: str,
    config: CheapFilterConfig,
    scorer: OllamaRowLogOddsScorer | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], CheapFilterMetrics]:
    """Score every row that survived structured filtering; drop only the ones
    a Beta-credible-bound reject threshold is confident are negative. Never
    accepts on its own -- everything not rejected goes on to the expensive
    batch join. Fails open: a scoring error means the row survives."""
    config.validate()
    scorer = scorer or OllamaRowLogOddsScorer(
        config.cheap_model, config.api_base, config.request_timeout
    )
    # Same dataset-id auto-detection convention already used by
    # project Trummer/baseline/trummer_join/operators.py's single_pass_join,
    # so this one implementation works unmodified for both the Amazon
    # (product_id/product_id) and IMDb (movie_id/tconst) trees.
    id_col_products = "product_id" if "product_id" in products_frame.columns else "movie_id"
    id_col_reviews = "product_id" if "product_id" in reviews_frame.columns else "tconst"

    metrics = CheapFilterMetrics()
    decisions: list[dict] = []
    scores_by_id: dict[str, float] = {}

    for record in reviews_frame.to_dict("records"):
        row_id = str(record.get(id_col_reviews, ""))
        review_text = str(record.get("review", ""))[: config.max_review_chars]
        metrics.rows_scored += 1
        metrics.cheap_calls += 1
        started = time.perf_counter()
        try:
            score = scorer.score(review_text, predicate)
        except Exception:
            metrics.cheap_seconds += time.perf_counter() - started
            metrics.cheap_failures += 1
            decisions.append({id_col_reviews: row_id, "score": None, "decision": "survive_error"})
            continue
        metrics.cheap_seconds += time.perf_counter() - started
        scores_by_id[row_id] = float(score)

    if config.manual_reject_threshold is not None:
        reject_threshold = config.manual_reject_threshold
        precision_lower = 1.0
        metrics.calibration_mode = "manual"
        metrics.calibration_activated = True
    elif config.ground_truth_ids is not None and scores_by_id:
        scores = list(scores_by_id.values())
        labels = [
            1 if row_id in config.ground_truth_ids else 0
            for row_id in scores_by_id
        ]
        result = fit_reject_threshold(
            scores,
            labels,
            precision_target=config.cascade_target,
            credible_level=config.credible_level,
        )
        reject_threshold = result.threshold
        precision_lower = result.precision_lower
        metrics.calibration_mode = "ground_truth"
        metrics.calibration_activated = math.isfinite(reject_threshold)
    else:
        reject_threshold = -math.inf
        precision_lower = 0.0
        metrics.calibration_mode = "disabled_no_ground_truth"
        metrics.calibration_activated = False

    metrics.reject_threshold = reject_threshold if math.isfinite(reject_threshold) else None
    metrics.reject_precision_lower = precision_lower

    survivor_ids: set[str] = set()
    for row_id, score in scores_by_id.items():
        if score <= reject_threshold:
            metrics.rows_rejected += 1
            decisions.append({id_col_reviews: row_id, "score": score, "decision": "reject"})
        else:
            survivor_ids.add(row_id)
            decisions.append({id_col_reviews: row_id, "score": score, "decision": "survive"})
    # Rows whose cheap scoring failed already got a "survive_error" decision
    # above and must still count as survivors.
    survivor_ids.update(
        str(item[id_col_reviews]) for item in decisions if item["decision"] == "survive_error"
    )
    metrics.rows_survived = len(survivor_ids)

    survivor_products = products_frame[
        products_frame[id_col_products].astype(str).isin(survivor_ids)
    ].reset_index(drop=True)
    survivor_reviews = reviews_frame[
        reviews_frame[id_col_reviews].astype(str).isin(survivor_ids)
    ].reset_index(drop=True)

    return survivor_products, survivor_reviews, decisions, metrics
