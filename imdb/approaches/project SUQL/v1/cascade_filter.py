from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np

from semantic_dict_context import semantic_guideline

from band_fit import DEFAULT_STRATA, fit_band, stratified_sample
from profiler import fit_cascade_threshold
from scorer import OllamaLogOddsScorer


DEFAULT_API_BASE = os.environ.get("SUQL_API_BASE", "http://localhost:11434")
DEFAULT_CHEAP_MODEL = os.environ.get("SUQL_CHEAP_MODEL", "ollama/gemma4:e2b")
DEFAULT_EXPENSIVE_MODEL = os.environ.get(
    "SUQL_EXPENSIVE_MODEL",
    os.environ.get("SUQL_MODEL", "ollama/gemma4:e4b"),
)
DEFAULT_CHEAP_ACCEPT_FLOOR = float(os.environ.get("SUQL_CHEAP_ACCEPT_FLOOR", "4.0"))
DEFAULT_CHEAP_MIN_DECISION_RATE = float(os.environ.get("SUQL_CHEAP_MIN_DECISION_RATE", "0.3"))
DEFAULT_CHEAP_MIN_PROBES = int(os.environ.get("SUQL_CHEAP_MIN_PROBES", "5"))
DEFAULT_CASCADE_TARGET = float(os.environ.get("SUQL_CASCADE_TARGET", "0.8"))
DEFAULT_CALIBRATION_BUDGET = int(os.environ.get("SUQL_CALIBRATION_BUDGET", "20"))
DEFAULT_CREDIBLE_LEVEL = float(os.environ.get("SUQL_CREDIBLE_LEVEL", "0.9"))
DEFAULT_CALIBRATION_STRATA = int(os.environ.get("SUQL_CALIBRATION_STRATA", str(DEFAULT_STRATA)))
DEFAULT_MANUAL_CONFIDENCE_THRESHOLD = os.environ.get("SUQL_MANUAL_CONFIDENCE_THRESHOLD")
DEFAULT_REQUEST_RETRIES = int(os.environ.get("SUQL_REQUEST_RETRIES", "3"))
EXPENSIVE_THINK = os.environ.get("EXPENSIVE_THINK", "0") == "1"
EXPENSIVE_NUM_PREDICT = int(
    os.environ.get("EXPENSIVE_NUM_PREDICT", "512" if EXPENSIVE_THINK else "96")
)


def parse_disabled_questions(value: str | None) -> set[str]:
    if not value:
        return set()
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        parsed = value.split("||")
    if isinstance(parsed, str):
        parsed = [parsed]
    return {str(item).strip() for item in parsed if str(item).strip()}


def litellm_model_name(model: str) -> str:
    """Return the model name expected by LiteLLM's Ollama provider."""
    return model if model.startswith("ollama/") else f"ollama/{model}"


def ollama_model_name(model: str) -> str:
    """Return the model name expected by Ollama's native API."""
    return model.removeprefix("ollama/")

ANSWER_SYSTEM = """You are the final recall-first semantic filter for movie reviews.

Decide whether the review provides evidence that the movie satisfies the question.

Recall is the primary objective, but do not accept every review:
- Answer YES for any concrete review-specific support, including direct wording,
  synonyms, described examples, and reasonable implications.
- Give borderline evidence the benefit of the doubt when it specifically bears on
  the condition; exact wording is unnecessary.
- This is evidence retrieval, not sentiment scoring: an explicit mention still
  counts when it is negated, qualified, quoted, or used critically.
- Answer NO when support is absent, merely generic, based only on genre/topic, or
  never discusses the condition itself. Lack of contradiction alone is not evidence.

Return exactly YES or NO. Do not include reasoning, punctuation, or extra text."""

@dataclass
class CascadeStats:
    cache_hits: int = 0
    cache_misses: int = 0
    cheap_score_calls: int = 0
    cheap_score_failures: int = 0
    cheap_seconds: float = 0.0
    cheap_early_accept: int = 0
    cheap_early_reject: int = 0
    cheap_skipped: int = 0
    cheap_disabled: int = 0
    calibration_candidates: int = 0
    calibration_expensive_calls: int = 0
    calibration_expensive_accepts: int = 0
    calibration_agreement: float = 0.0
    calibration_mode: str = ""
    calibration_labelled_count: int = 0
    calibration_accept_precision_lower: float = 0.0
    calibration_reject_precision_lower: float = 0.0
    calibration_reject_recall_lower: float = 0.0
    calibration_reused_labels: int = 0
    calibration_activated: bool = False
    learned_accept_threshold: float | None = None
    learned_reject_threshold: float | None = None
    expensive_full_calls: int = 0
    expensive_seconds: float = 0.0
    cheap_score_failure_reasons: dict[str, int] = field(default_factory=dict)
    model_usage_by_question: dict[str, dict[str, object]] = field(default_factory=dict)

    def usage_for(self, question: str, cheap_model: str, expensive_model: str) -> dict[str, object]:
        usage = self.model_usage_by_question.setdefault(
            question,
            {
                "cheap_model": cheap_model,
                "expensive_model": expensive_model,
                "cache_hits": 0,
                "cache_misses": 0,
                "cheap_score_calls": 0,
                "cheap_score_failures": 0,
                "cheap_seconds": 0.0,
                "cheap_early_accept": 0,
                "cheap_early_reject": 0,
                "cheap_skipped": 0,
                "cheap_disabled": 0,
                "no_model_early_reject": 0,
                "calibration_candidates": 0,
                "calibration_expensive_calls": 0,
                "calibration_expensive_accepts": 0,
                "calibration_agreement": 0.0,
                "calibration_mode": "",
                "calibration_labelled_count": 0,
                "calibration_accept_precision_lower": 0.0,
                "calibration_reject_precision_lower": 0.0,
                "calibration_reject_recall_lower": 0.0,
                "calibration_reused_labels": 0,
                "calibration_estimator": "",
                "cheap_score_mode": "",
                "calibration_strata_sizes": [],
                "calibration_strata_labelled": [],
                "calibration_strata_positives": [],
                "calibration_activated": False,
                "ground_truth_calibration_skipped_reason": "",
                "cascade_target": None,
                "calibration_budget": None,
                "credible_level": None,
                "learned_confidence_threshold": None,
                "routing_confidence_threshold": None,
                "learned_accept_threshold": None,
                "learned_reject_threshold": None,
                "manual_confidence_threshold": None,
                "expensive_full_calls": 0,
                "expensive_seconds": 0.0,
                "cheap_accept_floor": None,
                "cheap_min_decision_rate": None,
                "cheap_min_probes": None,
                "cheap_skip_reason": "",
                "cheap_score_failure_reasons": {},
            },
        )
        usage["cheap_model"] = cheap_model
        usage["expensive_model"] = expensive_model
        return usage

    def record_cheap_score_failure(self, question: str, cheap_model: str, expensive_model: str, exc: Exception) -> None:
        reason = f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
        reason = reason[:500]
        self.cheap_score_failure_reasons[reason] = self.cheap_score_failure_reasons.get(reason, 0) + 1

        usage = self.usage_for(question, cheap_model, expensive_model)
        reasons = usage.setdefault("cheap_score_failure_reasons", {})
        if isinstance(reasons, dict):
            reasons[reason] = int(reasons.get(reason, 0)) + 1

    def to_json_dict(self) -> dict:
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cheap_score_calls": self.cheap_score_calls,
            "cheap_score_failures": self.cheap_score_failures,
            "cheap_seconds": self.cheap_seconds,
            "cheap_early_accept": self.cheap_early_accept,
            "cheap_early_reject": self.cheap_early_reject,
            "cheap_skipped": self.cheap_skipped,
            "cheap_disabled": self.cheap_disabled,
            "calibration_candidates": self.calibration_candidates,
            "calibration_expensive_calls": self.calibration_expensive_calls,
            "calibration_expensive_accepts": self.calibration_expensive_accepts,
            "calibration_agreement": self.calibration_agreement,
            "calibration_mode": self.calibration_mode,
            "calibration_labelled_count": self.calibration_labelled_count,
            "calibration_accept_precision_lower": self.calibration_accept_precision_lower,
            "calibration_reject_precision_lower": self.calibration_reject_precision_lower,
            "calibration_reject_recall_lower": self.calibration_reject_recall_lower,
            "calibration_reused_labels": self.calibration_reused_labels,
            "calibration_activated": self.calibration_activated,
            "learned_accept_threshold": self.learned_accept_threshold,
            "learned_reject_threshold": self.learned_reject_threshold,
            "expensive_full_calls": self.expensive_full_calls,
            "expensive_seconds": self.expensive_seconds,
            "cheap_score_failure_reasons": self.cheap_score_failure_reasons,
            "model_usage_by_question": self.model_usage_by_question,
        }


@dataclass
class CascadeAnswerFilter:
    thresholds_path: str | Path | None = None
    cheap_model: str = DEFAULT_CHEAP_MODEL
    expensive_model: str = DEFAULT_EXPENSIVE_MODEL
    api_base: str = DEFAULT_API_BASE
    cheap_accept_floor: float = DEFAULT_CHEAP_ACCEPT_FLOOR
    cheap_min_decision_rate: float = DEFAULT_CHEAP_MIN_DECISION_RATE
    cheap_min_probes: int = DEFAULT_CHEAP_MIN_PROBES
    cascade_target: float = DEFAULT_CASCADE_TARGET
    calibration_budget: int = DEFAULT_CALIBRATION_BUDGET
    credible_level: float = DEFAULT_CREDIBLE_LEVEL
    calibration_strata: int = DEFAULT_CALIBRATION_STRATA
    ground_truth_ids: frozenset[str] | None = None
    manual_confidence_threshold: float | None = (
        float(DEFAULT_MANUAL_CONFIDENCE_THRESHOLD)
        if DEFAULT_MANUAL_CONFIDENCE_THRESHOLD not in (None, "")
        else None
    )
    cheap_disabled_questions: set[str] | list[str] | tuple[str, ...] | None = None
    timeout: float = 120.0
    request_retries: int = DEFAULT_REQUEST_RETRIES
    stats: CascadeStats = field(default_factory=CascadeStats)

    def __post_init__(self) -> None:
        self.api_base = self.api_base.rstrip("/")
        self.cheap_model = litellm_model_name(self.cheap_model)
        self.expensive_model = litellm_model_name(self.expensive_model)
        self.cheap_accept_floor = float(self.cheap_accept_floor)
        self.cheap_min_decision_rate = float(self.cheap_min_decision_rate)
        self.cheap_min_probes = int(self.cheap_min_probes)
        self.cascade_target = float(self.cascade_target)
        self.calibration_budget = int(self.calibration_budget)
        self.credible_level = float(self.credible_level)
        self.calibration_strata = int(self.calibration_strata)
        self.request_retries = max(1, int(self.request_retries))
        if not 0.0 < self.cascade_target <= 1.0:
            raise ValueError("cascade_target must be in (0, 1]")
        if self.calibration_budget < 0:
            raise ValueError("calibration_budget must be non-negative")
        if not 0.0 < self.credible_level < 1.0:
            raise ValueError("credible_level must be in (0, 1)")
        if self.calibration_strata < 2:
            raise ValueError("calibration_strata must be at least 2")
        if self.ground_truth_ids is not None:
            self.ground_truth_ids = frozenset(str(x) for x in self.ground_truth_ids)
        if self.manual_confidence_threshold is not None:
            self.manual_confidence_threshold = float(self.manual_confidence_threshold)
            if self.manual_confidence_threshold < 0:
                raise ValueError("manual_confidence_threshold must be non-negative")
        self.cheap_disabled_questions = {
            str(question).strip()
            for question in (self.cheap_disabled_questions or set())
            if str(question).strip()
        }
        self.cheap_ollama_model = ollama_model_name(self.cheap_model)
        self.expensive_ollama_model = ollama_model_name(self.expensive_model)
        self.cheap_scorer = OllamaLogOddsScorer(
            model=self.cheap_model,
            api_base=self.api_base,
            timeout=self.timeout,
        )
        self.cache: dict[str, str] = {}

    @staticmethod
    def _cache_key(review_text: str, question: str) -> str:
        raw = f"{review_text}|||{question}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _record_cheap_skip(self, usage: dict[str, object]) -> None:
        self.stats.cheap_skipped += 1
        usage["cheap_skipped"] = int(usage["cheap_skipped"]) + 1

    def _record_cheap_disabled(self, usage: dict[str, object]) -> None:
        self.stats.cheap_disabled += 1
        usage["cheap_disabled"] = int(usage["cheap_disabled"]) + 1
        usage["cheap_skip_reason"] = "disabled for this question"

    def answer(self, review_text: str, question: str) -> str:
        return self.answer_batch([review_text], question)[0]

    def answer_batch(
        self,
        review_texts: list[str],
        question: str,
        *,
        row_ids: list[str | None] | None = None,
    ) -> list[str]:
        usage = self.stats.usage_for(question, self.cheap_model, self.expensive_model)
        usage["cheap_accept_floor"] = self.cheap_accept_floor
        usage["cheap_min_decision_rate"] = self.cheap_min_decision_rate
        usage["cheap_min_probes"] = self.cheap_min_probes
        usage["cascade_target"] = self.cascade_target
        usage["calibration_budget"] = self.calibration_budget
        usage["credible_level"] = self.credible_level
        usage["manual_confidence_threshold"] = self.manual_confidence_threshold
        usage["cheap_disabled_for_question"] = question in self.cheap_disabled_questions
        results: list[str | None] = [None] * len(review_texts)
        scored: list[tuple[int, str, float, str | None]] = []
        ids: list[str | None] = (
            list(row_ids) if row_ids is not None and len(row_ids) == len(review_texts)
            else [None] * len(review_texts)
        )

        for index, raw_text in enumerate(review_texts):
            review_text = "" if raw_text is None else str(raw_text)
            key = self._cache_key(review_text, question)
            if key in self.cache:
                self.stats.cache_hits += 1
                usage["cache_hits"] = int(usage["cache_hits"]) + 1
                results[index] = self.cache[key]
                continue

            self.stats.cache_misses += 1
            usage["cache_misses"] = int(usage["cache_misses"]) + 1
            if not review_text or len(review_text.strip()) < 10:
                self.stats.cheap_early_reject += 1
                usage["cheap_early_reject"] = int(usage["cheap_early_reject"]) + 1
                usage["no_model_early_reject"] = int(usage["no_model_early_reject"]) + 1
                self.cache[key] = "No"
                results[index] = "No"
                continue

            if question in self.cheap_disabled_questions:
                self._record_cheap_disabled(usage)
                result = self.expensive_answer(review_text, question)
                self.cache[key] = result
                results[index] = result
                continue

            self.stats.cheap_score_calls += 1
            usage["cheap_score_calls"] = int(usage["cheap_score_calls"]) + 1
            started = time.perf_counter()
            try:
                score = float(self.cheap_scorer.score(review_text, question))
            except Exception as exc:
                elapsed = time.perf_counter() - started
                self.stats.cheap_seconds += elapsed
                usage["cheap_seconds"] = float(usage["cheap_seconds"]) + elapsed
                self.stats.cheap_score_failures += 1
                usage["cheap_score_failures"] = int(usage["cheap_score_failures"]) + 1
                self.stats.record_cheap_score_failure(question, self.cheap_model, self.expensive_model, exc)
                result = self.expensive_answer(review_text, question)
                self.cache[key] = result
                results[index] = result
                continue
            elapsed = time.perf_counter() - started
            self.stats.cheap_seconds += elapsed
            usage["cheap_seconds"] = float(usage["cheap_seconds"]) + elapsed

            scored.append((index, review_text, score, ids[index]))

        usage["cheap_score_mode"] = getattr(self.cheap_scorer, "score_mode", "")
        calibration_answers: dict[int, str] = {}
        if self.manual_confidence_threshold is None:
            accept_threshold, reject_threshold = self._learn_confidence_threshold(
                scored, question, usage, calibration_answers
            )
        else:
            accept_threshold, reject_threshold = (
                self.manual_confidence_threshold,
                -self.manual_confidence_threshold,
            )
            usage["calibration_mode"] = "manual"
            self.stats.calibration_mode = "manual"
            self.stats.calibration_activated = True
        usage["learned_accept_threshold"] = accept_threshold
        usage["learned_reject_threshold"] = reject_threshold
        # Kept for backward compatibility with anything still reading the old
        # single-threshold keys; mirrors the accept-side value.
        usage["learned_confidence_threshold"] = (
            accept_threshold if math.isfinite(accept_threshold) else None
        )
        usage["routing_confidence_threshold"] = accept_threshold
        self.stats.learned_accept_threshold = (
            accept_threshold if math.isfinite(accept_threshold) else None
        )
        self.stats.learned_reject_threshold = (
            reject_threshold if math.isfinite(reject_threshold) else None
        )
        self.stats.calibration_mode = str(usage.get("calibration_mode") or self.stats.calibration_mode)
        self.stats.calibration_labelled_count = int(usage.get("calibration_labelled_count") or 0)
        self.stats.calibration_accept_precision_lower = float(
            usage.get("calibration_accept_precision_lower") or 0.0
        )
        self.stats.calibration_reject_precision_lower = float(
            usage.get("calibration_reject_precision_lower") or 0.0
        )
        self.stats.calibration_activated = bool(
            usage.get("calibration_activated", self.stats.calibration_activated)
        )

        # Rows the oracle already labelled during calibration carry the expensive
        # model's answer; use it instead of routing (and paying for) them again.
        for index, review_text, _score, _row_id in scored:
            answer = calibration_answers.get(index)
            if answer is not None:
                self.cache[self._cache_key(review_text, question)] = answer
                results[index] = answer
        usage["calibration_reused_labels"] = int(usage["calibration_reused_labels"]) + len(calibration_answers)
        self.stats.calibration_reused_labels += len(calibration_answers)

        for index, review_text, score, _row_id in scored:
            if results[index] is not None:
                continue
            key = self._cache_key(review_text, question)
            if score >= accept_threshold:
                self.stats.cheap_early_accept += 1
                usage["cheap_early_accept"] = int(usage["cheap_early_accept"]) + 1
                self.cache[key] = "Yes"
                results[index] = "Yes"
            elif score <= reject_threshold:
                self.stats.cheap_early_reject += 1
                usage["cheap_early_reject"] = int(usage["cheap_early_reject"]) + 1
                self.cache[key] = "No"
                results[index] = "No"
            else:
                result = self.expensive_answer(review_text, question)
                self.cache[key] = result
                results[index] = result

        return [str(result or "No") for result in results]

    def _learn_confidence_threshold(
        self,
        scored: list[tuple[int, str, float, str | None]],
        question: str,
        usage: dict[str, object],
        calibration_answers: dict[int, str] | None = None,
    ) -> tuple[float, float]:
        use_ground_truth = self.ground_truth_ids is not None and all(
            row_id for *_rest, row_id in scored
        )
        if use_ground_truth:
            usage["calibration_mode"] = "ground_truth"
            return self._learn_threshold_from_ground_truth(scored, usage)

        if self.ground_truth_ids is not None:
            # ground_truth_ids configured but some rows lack an id -- never mix
            # modes per-row, fall back to the oracle path entirely.
            usage["ground_truth_calibration_skipped_reason"] = "row_ids unavailable"
        usage["calibration_mode"] = "oracle"
        return self._learn_threshold_from_oracle(scored, question, usage, calibration_answers)

    def _learn_threshold_from_oracle(
        self,
        scored: list[tuple[int, str, float, str | None]],
        question: str,
        usage: dict[str, object],
        calibration_answers: dict[int, str] | None = None,
    ) -> tuple[float, float]:
        """Fit the accept/reject band against expensive-model labels.

        Follows paper sec:cascade-math: the reject side is certified by a Beta
        credible lower bound on *recall* (tau_-), the accept side on *precision*
        (tau_+). The labelled sample is stratified by cheap score and the
        stratified posterior in band_fit.py turns it into those bounds. Every
        labelled row keeps its oracle answer (``calibration_answers``) so the
        expensive calls spent here are not repeated.
        """
        usage["calibration_estimator"] = "stratified_beta_mc"
        pool_scores = [score for _index, _text, score, _row_id in scored]
        raw_seed = os.environ.get("CALIBRATION_SEED", "").strip()
        rng = random.Random(raw_seed) if raw_seed and raw_seed.lower() != "none" else None
        picks = stratified_sample(pool_scores, self.calibration_budget, self.calibration_strata, rng)
        if not picks:
            usage["calibration_agreement"] = 0.0
            usage["calibration_activated"] = False
            self.stats.calibration_agreement = 0.0
            return math.inf, -math.inf

        labels: dict[int, int] = {}
        self.stats.calibration_candidates += len(picks)
        usage["calibration_candidates"] = int(usage["calibration_candidates"]) + len(picks)
        for pick in picks:
            index, review_text, _score, _row_id = scored[pick]
            self.stats.calibration_expensive_calls += 1
            usage["calibration_expensive_calls"] = int(usage["calibration_expensive_calls"]) + 1
            oracle = self.expensive_answer(review_text, question)
            oracle_label = _normalised_yes_no(oracle)
            if oracle_label is None:
                continue
            if oracle_label:
                self.stats.calibration_expensive_accepts += 1
                usage["calibration_expensive_accepts"] = int(usage["calibration_expensive_accepts"]) + 1
            labels[pick] = 1 if oracle_label else 0
            if calibration_answers is not None:
                calibration_answers[index] = "Yes" if oracle_label else "No"
        usage["calibration_labelled_count"] = len(labels)

        if not labels:
            usage["calibration_agreement"] = 0.0
            usage["calibration_activated"] = False
            self.stats.calibration_agreement = 0.0
            return math.inf, -math.inf

        fit = fit_band(
            pool_scores,
            labels,
            quality_target=self.cascade_target,
            credible_level=self.credible_level,
            n_strata=self.calibration_strata,
        )
        usage["calibration_accept_precision_lower"] = fit.accept_precision_lower
        usage["calibration_reject_recall_lower"] = fit.reject_recall_lower
        usage["calibration_strata_sizes"] = list(fit.strata_sizes)
        usage["calibration_strata_labelled"] = list(fit.strata_labelled)
        usage["calibration_strata_positives"] = list(fit.strata_positives)
        self.stats.calibration_accept_precision_lower = fit.accept_precision_lower
        self.stats.calibration_reject_recall_lower = fit.reject_recall_lower

        selected = [
            (pool_scores[pick] >= fit.accept_threshold) or (pool_scores[pick] <= fit.reject_threshold)
            for pick in labels
        ]
        selected_count = sum(selected)
        agreed = sum(
            1
            for (pick, label), is_selected in zip(labels.items(), selected)
            if is_selected and int(pool_scores[pick] >= 0) == label
        )
        agreement = agreed / selected_count if selected_count else 0.0
        self.stats.calibration_agreement = agreement
        usage["calibration_agreement"] = agreement
        usage["calibration_activated"] = fit.activated
        return fit.accept_threshold, fit.reject_threshold

    def _learn_threshold_from_ground_truth(
        self,
        scored: list[tuple[int, str, float, str | None]],
        usage: dict[str, object],
    ) -> tuple[float, float]:
        assert self.ground_truth_ids is not None
        scores = [score for _index, _review_text, score, _row_id in scored]
        labels = [
            1 if row_id in self.ground_truth_ids else 0
            for _index, _review_text, _score, row_id in scored
        ]
        self.stats.calibration_candidates += len(scores)
        usage["calibration_candidates"] = int(usage["calibration_candidates"]) + len(scores)
        usage["calibration_labelled_count"] = len(scores)

        if not scores:
            usage["calibration_agreement"] = 0.0
            usage["calibration_activated"] = False
            self.stats.calibration_agreement = 0.0
            return math.inf, -math.inf

        result = fit_cascade_threshold(
            "",
            np.asarray(scores, dtype=float),
            np.asarray(labels, dtype=int),
            accept_precision_target=self.cascade_target,
            reject_precision_target=self.cascade_target,
            credible_level=self.credible_level,
        )
        usage["calibration_accept_precision_lower"] = result.accept_precision_lower
        usage["calibration_reject_precision_lower"] = result.reject_precision_lower

        selected = [
            (score >= result.cheap_accept_threshold) or (score <= result.cheap_reject_threshold)
            for score in scores
        ]
        selected_count = sum(selected)
        agreed = sum(
            1
            for score, label, is_selected in zip(scores, labels, selected)
            if is_selected and int(score >= 0) == label
        )
        agreement = agreed / selected_count if selected_count else 0.0
        self.stats.calibration_agreement = agreement
        usage["calibration_agreement"] = agreement
        usage["calibration_activated"] = (
            result.cheap_accept_threshold < math.inf or result.cheap_reject_threshold > -math.inf
        )
        return result.cheap_accept_threshold, result.cheap_reject_threshold

    def expensive_answer(self, review_text: str, question: str) -> str:
        usage = self.stats.usage_for(question, self.cheap_model, self.expensive_model)
        self.stats.expensive_full_calls += 1
        usage["expensive_full_calls"] = int(usage["expensive_full_calls"]) + 1
        guidance = semantic_guideline(question)
        user_prompt = (
            (f"{guidance}\n\n" if guidance else "")
            + f"Review:\n{review_text[:6000]}\n\nQuestion: {question}"
        )
        payload = {
            "model": self.expensive_ollama_model,
            "messages": [
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": EXPENSIVE_THINK,
            "keep_alive": "30m",
            "options": {
                "temperature": 0,
                "num_predict": EXPENSIVE_NUM_PREDICT,
                "num_ctx": 8192,
            },
        }
        trace = os.environ.get("PIPELINE_TRACE", "0") == "1"
        if trace:
            print(f"[TRACE SUQL EXPENSIVE PROMPT] {payload!r}", flush=True)
        for attempt in range(2):
            started = time.perf_counter()
            try:
                response = _post_with_retries(
                    f"{self.api_base}/api/chat", payload,
                    timeout=self.timeout, retries=self.request_retries,
                )
                result = extract_text(response.json()).strip()
            except Exception:
                result = ""
            finally:
                elapsed = time.perf_counter() - started
                self.stats.expensive_seconds += elapsed
                usage["expensive_seconds"] = float(usage["expensive_seconds"]) + elapsed
            if trace:
                print(f"[TRACE SUQL EXPENSIVE OUTPUT] {result!r}", flush=True)
            decision = _parse_expensive_decision(result)
            if decision is not None:
                return decision
            payload["messages"][-1]["content"] = (
                user_prompt + "\n\nYour previous response was invalid. Reply with exactly YES or NO."
            )
        return "No"


def _parse_expensive_decision(result: str) -> str | None:
    try:
        structured = json.loads(result)
        if isinstance(structured, dict):
            result = str(structured.get("decision", structured.get("answer", result)))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    match = re.search(r"\b(uncertain|yes|no)\b", result, re.IGNORECASE)
    if not match or match.group(1).lower() == "uncertain":
        return None
    return match.group(1).capitalize()


def _post_with_retries(
    url: str,
    payload: dict,
    *,
    timeout: float,
    retries: int,
) -> httpx.Response:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = httpx.post(url, json=payload, timeout=timeout)
            if response.status_code < 500:
                response.raise_for_status()
                return response
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            last_error = RuntimeError(_http_error_message(exc))
            if exc.response.status_code < 500 or attempt == retries:
                raise last_error from exc
        except httpx.HTTPError as exc:
            last_error = exc
            if attempt == retries:
                raise
        time.sleep(min(30, 5 * attempt))
    if last_error is not None:
        raise last_error
    raise RuntimeError("Ollama request failed without an exception")


def _http_error_message(exc: httpx.HTTPStatusError) -> str:
    body = exc.response.text.strip().replace("\n", " ")
    if len(body) > 1000:
        body = body[:1000] + "..."
    suffix = f": {body}" if body else ""
    return (
        f"Ollama request failed with HTTP {exc.response.status_code} "
        f"for {exc.request.url}{suffix}"
    )


def _normalised_yes_no(value: str) -> bool | None:
    low = value.strip().lower().rstrip(".")
    if low == "yes":
        return True
    if low == "no":
        return False
    return None


def extract_text(payload: dict) -> str:
    if "response" in payload:
        return str(payload["response"])
    message = payload.get("message")
    if isinstance(message, dict):
        return str(message.get("content", ""))
    choices = payload.get("choices") or []
    if choices:
        choice = choices[0]
        if "text" in choice:
            return str(choice["text"])
        message = choice.get("message")
        if isinstance(message, dict):
            return str(message.get("content", ""))
    return ""
