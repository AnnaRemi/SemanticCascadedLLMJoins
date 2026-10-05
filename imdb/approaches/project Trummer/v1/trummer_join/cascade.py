from __future__ import annotations

import json
import math
import os
import random
import re
import time
import urllib.request
from dataclasses import asdict, dataclass
from typing import Callable, Iterable


PAIR_LABEL_RE = re.compile(r"\b(?:candidate|pair)[_ #:-]*(\d+)\b", re.IGNORECASE)
PLAIN_NUMBER_RE = re.compile(r"^\s*(\d+)\s*$")

from .semantic_dict_context import semantic_guideline
from .band_fit import DEFAULT_STRATA, fit_band, stratified_sample

CHEAP_EVIDENCE_INSTRUCTIONS = """Act as a high-recall first-pass semantic filter.

Answer YES for direct evidence, synonyms, described examples, or reasonable
implications. Answer UNCERTAIN when evidence is condition-specific but too weak or
ambiguous for a reliable YES. Answer NO only when review-specific support is absent,
generic, based only on genre/topic, or never discusses the predicate itself. Lack of contradiction is not
evidence. Count explicit mentions even when negated, qualified, quoted, or critical.
Prefer UNCERTAIN over NO when plausible condition-specific evidence exists."""

EXPENSIVE_EVIDENCE_INSTRUCTIONS = """Act as the final recall-first semantic filter.

Answer YES for any concrete review-specific support, including direct wording,
synonyms, described examples, and reasonable implications. Give borderline evidence
the benefit of the doubt when it specifically bears on the predicate. Answer NO when
support is absent, generic, based only on genre/topic, or never discusses the predicate itself. Lack of
contradiction alone is not evidence. This is evidence retrieval, not sentiment
scoring: count explicit mentions even when negated, qualified, quoted, or critical."""

EXPENSIVE_THINK = os.environ.get("EXPENSIVE_THINK", "0") == "1"
EXPENSIVE_NUM_PREDICT = int(
    os.environ.get("EXPENSIVE_NUM_PREDICT", "512" if EXPENSIVE_THINK else "128")
)


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    movie: dict[str, str]
    review: dict[str, str]


@dataclass(frozen=True)
class Decision:
    candidate_id: int
    movie_id: str
    tconst: str
    score: float | None
    route: str
    error: str = ""


@dataclass(frozen=True)
class CascadeConfig:
    api_base: str = "http://127.0.0.1:11434"
    cheap_model: str = "gemma4:e2b"
    expensive_model: str = "gemma4:e4b"
    cascade_target: float = 0.8
    calibration_budget: int = 20
    credible_level: float = 0.9
    calibration_strata: int = DEFAULT_STRATA
    manual_confidence_threshold: float | None = None
    cheap_batch_size: int = 8
    expensive_batch_size: int = 32
    request_timeout: float = 600.0
    max_review_chars: int = 3500

    def validate(self) -> None:
        if not 0.0 < self.cascade_target <= 1.0:
            raise ValueError("cascade_target must be in (0, 1]")
        if self.calibration_budget < 0:
            raise ValueError("calibration_budget must be non-negative")
        if not 0.0 < self.credible_level < 1.0:
            raise ValueError("credible_level must be in (0, 1)")
        if self.calibration_strata < 2:
            raise ValueError("calibration_strata must be at least 2")
        if (
            self.manual_confidence_threshold is not None
            and self.manual_confidence_threshold < 0
        ):
            raise ValueError("manual_confidence_threshold must be non-negative")
        if self.cheap_batch_size < 1:
            raise ValueError("cheap_batch_size must be positive")
        if self.expensive_batch_size < 1:
            raise ValueError("expensive_batch_size must be positive")
        if self.expensive_batch_size <= self.cheap_batch_size:
            raise ValueError(
                "expensive_batch_size must exceed cheap_batch_size so fallback "
                "calls can be coalesced across cheap batches"
            )


@dataclass
class CascadeMetrics:
    input_movies: int = 0
    input_reviews: int = 0
    candidate_pairs: int = 0
    cheap_batches: int = 0
    cheap_calls: int = 0
    cheap_failures: int = 0
    cheap_failure_candidates: int = 0
    cheap_early_accepts: int = 0
    cheap_early_rejects: int = 0
    calibration_candidates: int = 0
    calibration_expensive_calls: int = 0
    calibration_expensive_accepts: int = 0
    calibration_agreement: float = 0.0
    calibration_accept_precision_lower: float = 0.0
    calibration_reject_precision_lower: float = 0.0
    calibration_reject_recall_lower: float = 0.0
    calibration_reused_labels: int = 0
    calibration_activated: bool = False
    learned_confidence_threshold: float | None = None
    routing_confidence_threshold: float | None = None
    learned_accept_threshold: float | None = None
    learned_reject_threshold: float | None = None
    manual_confidence_threshold: float | None = None
    expensive_candidates: int = 0
    expensive_calls: int = 0
    fallback_expensive_calls: int = 0
    expensive_failures: int = 0
    expensive_undecided: int = 0
    expensive_accepts: int = 0
    nonempty_fallback_rows: int = 0
    cheap_seconds: float = 0.0
    expensive_seconds: float = 0.0
    cheap_time_percent: float = 0.0
    expensive_time_percent: float = 0.0
    elapsed_seconds: float = 0.0


class OllamaBatchBinaryScorer:
    """Classify one candidate batch in one cheap-model request.

    The model's plain-text hard decisions are represented as
    -2/+2 so the cascade can derive proxy labels and confidence values.
    """

    def __init__(self, config: CascadeConfig) -> None:
        self.config = config
        self.api_base = config.api_base.rstrip("/")

    def score_batch(
        self,
        candidates: list[Candidate],
        predicate: str,
    ) -> dict[int, float]:
        if not candidates:
            return {}
        prompt = _cheap_batch_prompt(
            candidates,
            predicate,
            self.config.max_review_chars,
        )
        ids = [candidate.candidate_id for candidate in candidates]
        request_payload = {
            "model": _plain_model(self.config.cheap_model),
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {
                "temperature": 0,
                "num_predict": max(128, 32 + 24 * len(ids)),
            "num_ctx": 8192,
            },
        }
        trace = os.environ.get("PIPELINE_TRACE", "0") == "1"
        if trace:
            print(f"[TRACE V3_2 CHEAP PROMPT]\n{prompt}\n[TRACE END]", flush=True)
        payload = _post_json(
            f"{self.api_base}/api/chat",
            request_payload,
            self.config.request_timeout,
        )
        if trace:
            print(f"[TRACE V3_2 CHEAP OUTPUT] {payload!r}", flush=True)
        return _parse_batch_scores(payload, set(ids))


class OllamaExpensiveClassifier:
    """Classify uncertain candidates in coalesced expensive-model batches."""

    def __init__(self, config: CascadeConfig) -> None:
        self.config = config
        self.api_base = config.api_base.rstrip("/")

    def classify(self, candidates: list[Candidate], predicate: str) -> set[int]:
        if not candidates:
            return set()
        prompt = _expensive_batch_prompt(
            candidates,
            predicate,
            self.config.max_review_chars,
        )
        valid = {candidate.candidate_id for candidate in candidates}
        request_payload = {
            "model": _plain_model(self.config.expensive_model),
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": EXPENSIVE_THINK,
            "options": {
                "temperature": 0,
                "num_predict": max(EXPENSIVE_NUM_PREDICT, 32 + 24 * len(valid)),
                "num_ctx": 8192,
            },
        }
        trace = os.environ.get("PIPELINE_TRACE", "0") == "1"
        if trace:
            print(f"[TRACE V3_2 EXPENSIVE PROMPT]\n{prompt}\n[TRACE END]", flush=True)
        payload = _post_json(
            f"{self.api_base}/api/chat",
            request_payload,
            self.config.request_timeout,
        )
        answer = str(payload.get("message", {}).get("content", ""))
        if trace:
            print(f"[TRACE V3_2 EXPENSIVE OUTPUT] {payload!r}", flush=True)
        decisions = _parse_expensive_decisions(answer, valid)
        if set(decisions) != valid:
            missing = sorted(valid - set(decisions))
            raise ValueError(f"expensive batch response omitted candidate IDs: {missing}")
        return {candidate_id for candidate_id, label in decisions.items() if label == "yes"}


class CascadeJoin:
    def __init__(
        self,
        config: CascadeConfig,
        cheap_score_batch: Callable[
            [list[Candidate], str], dict[int, float]
        ] | None = None,
        expensive_classify: Callable[
            [list[Candidate], str], set[int]
        ] | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.cheap_score_batch = (
            cheap_score_batch or OllamaBatchBinaryScorer(config).score_batch
        )
        self.expensive_classify = (
            expensive_classify or OllamaExpensiveClassifier(config).classify
        )

    def run(
        self,
        movies: list[dict[str, str]],
        reviews: list[dict[str, str]],
        predicate: str,
    ) -> tuple[list[dict[str, str]], list[Decision], CascadeMetrics]:
        started = time.perf_counter()
        candidates = exact_id_candidates(movies, reviews)
        metrics = CascadeMetrics(
            input_movies=len(movies),
            input_reviews=len(reviews),
            candidate_pairs=len(candidates),
        )
        decisions: list[Decision] = []
        accepted: list[Candidate] = []
        uncertain: list[Candidate] = []
        scored: list[tuple[Candidate, float | None, str]] = []

        cheap_batches = list(_blocks(candidates, self.config.cheap_batch_size))
        cheap_started = time.perf_counter()
        for batch_index, batch in enumerate(cheap_batches, 1):
            metrics.cheap_batches += 1
            metrics.cheap_calls += 1
            call_started = time.perf_counter()
            try:
                scores = self.cheap_score_batch(batch, predicate)
            except Exception as exc:
                metrics.cheap_seconds += time.perf_counter() - call_started
                metrics.cheap_failures += 1
                metrics.cheap_failure_candidates += len(batch)
                scored.extend((candidate, None, str(exc)) for candidate in batch)
                _print_progress(
                    "Batch-wise cascade cheap scoring",
                    batch_index,
                    len(cheap_batches),
                    cheap_started,
                )
                continue
            metrics.cheap_seconds += time.perf_counter() - call_started
            _print_progress(
                "Batch-wise cascade cheap scoring",
                batch_index,
                len(cheap_batches),
                cheap_started,
            )

            for candidate in batch:
                score = scores.get(candidate.candidate_id)
                if score is None:
                    metrics.cheap_failure_candidates += 1
                    scored.append(
                        (candidate, None, "cheap batch omitted candidate")
                    )
                else:
                    scored.append((candidate, float(score), ""))

        calibration_oracle: dict[int, bool] = {}
        if self.config.manual_confidence_threshold is None:
            accept_threshold, reject_threshold = self._learn_confidence_threshold(
                scored, predicate, metrics, calibration_oracle
            )
        else:
            accept_threshold = self.config.manual_confidence_threshold
            reject_threshold = -self.config.manual_confidence_threshold
            metrics.calibration_activated = True
        # Kept for backward compatibility with anything still reading the old
        # single-threshold keys; mirrors the accept-side value.
        metrics.learned_confidence_threshold = (
            accept_threshold if math.isfinite(accept_threshold) else None
        )
        metrics.routing_confidence_threshold = accept_threshold
        metrics.learned_accept_threshold = (
            accept_threshold if math.isfinite(accept_threshold) else None
        )
        metrics.learned_reject_threshold = (
            reject_threshold if math.isfinite(reject_threshold) else None
        )
        metrics.manual_confidence_threshold = self.config.manual_confidence_threshold

        metrics.calibration_reused_labels = len(calibration_oracle)
        for candidate, score, error in scored:
            oracle_accepts = calibration_oracle.get(candidate.candidate_id)
            if oracle_accepts is not None:
                # Already answered by the expensive model during calibration.
                if oracle_accepts:
                    accepted.append(candidate)
                decisions.append(
                    _decision(
                        candidate,
                        score,
                        "calibration_accept" if oracle_accepts else "calibration_reject",
                    )
                )
            elif score is None:
                metrics.expensive_candidates += 1
                uncertain.append(candidate)
                decisions.append(_decision(candidate, None, "expensive", error))
            elif score >= accept_threshold:
                metrics.cheap_early_accepts += 1
                accepted.append(candidate)
                decisions.append(_decision(candidate, score, "cheap_accept"))
            elif score <= reject_threshold:
                metrics.cheap_early_rejects += 1
                decisions.append(_decision(candidate, score, "cheap_reject"))
            else:
                metrics.expensive_candidates += 1
                uncertain.append(candidate)
                decisions.append(_decision(candidate, score, "expensive"))

        expensive_accepted: set[int] = set()
        fallback_batches = list(_blocks(uncertain, self.config.expensive_batch_size))
        fallback_started = time.perf_counter()
        for batch_index, batch in enumerate(fallback_batches, 1):
            batch_accepts, _decided = self._expensive_decide(batch, predicate, metrics)
            expensive_accepted.update(batch_accepts)
            _print_progress(
                "Batch-wise cascade expensive fallback",
                batch_index,
                len(fallback_batches),
                fallback_started,
            )

        by_id = {candidate.candidate_id: candidate for candidate in uncertain}
        accepted.extend(
            by_id[candidate_id]
            for candidate_id in sorted(expensive_accepted)
            if candidate_id in by_id
        )
        metrics.expensive_accepts = len(expensive_accepted & by_id.keys())
        metrics.elapsed_seconds = time.perf_counter() - started
        _set_time_percentages(metrics)

        cheap_accepted_ids = {
            decision.candidate_id
            for decision in decisions
            if decision.route == "cheap_accept"
        }
        rows = [
            joined_row(
                candidate,
                "cheap_accept"
                if candidate.candidate_id in cheap_accepted_ids
                else "expensive_accept",
            )
            for candidate in accepted
        ]
        return _deduplicate(rows), decisions, metrics

    def _expensive_decide(
        self,
        batch: list[Candidate],
        predicate: str,
        metrics: CascadeMetrics,
        calibration: bool = False,
    ) -> tuple[set[int], set[int]]:
        """Classify ``batch`` with the expensive model; returns (accepted ids, decided ids).

        A batch response can be unusable (the prompt overflowed the context window and
        the model omitted candidate ids, or the reply was malformed). Dropping the whole
        batch would silently discard every candidate in it, so a failed batch is split in
        two and each half retried, down to single candidates. Every attempt is counted as
        an expensive call and its time is charged, so retries are not free.
        """
        metrics.expensive_calls += 1
        if calibration:
            metrics.calibration_expensive_calls += 1
        else:
            metrics.fallback_expensive_calls += 1
        started = time.perf_counter()
        try:
            accepts = self.expensive_classify(batch, predicate)
        except Exception:
            metrics.expensive_seconds += time.perf_counter() - started
            metrics.expensive_failures += 1
            if len(batch) == 1:
                metrics.expensive_undecided += 1
                return set(), set()
            middle = len(batch) // 2
            left_accepts, left_decided = self._expensive_decide(
                batch[:middle], predicate, metrics, calibration
            )
            right_accepts, right_decided = self._expensive_decide(
                batch[middle:], predicate, metrics, calibration
            )
            return left_accepts | right_accepts, left_decided | right_decided
        metrics.expensive_seconds += time.perf_counter() - started
        ids = {candidate.candidate_id for candidate in batch}
        return accepts & ids, ids

    def _learn_confidence_threshold(
        self,
        scored: list[tuple[Candidate, float | None, str]],
        predicate: str,
        metrics: CascadeMetrics,
        calibration_oracle: dict[int, bool] | None = None,
    ) -> tuple[float, float]:
        """Fit the accept/reject band against expensive-model labels.

        Paper sec:cascade-math: the reject side is certified by a Beta credible
        lower bound on recall, the accept side on precision. The labelled sample
        is stratified by cheap score and band_fit.fit_band turns it into those
        bounds. Every labelled pair keeps its oracle answer
        (``calibration_oracle``), so the expensive call is not repeated.
        """
        usable = [(candidate, score) for candidate, score, _ in scored if score is not None]
        pool_scores = [score for _candidate, score in usable]
        picks = stratified_sample(
            pool_scores,
            self.config.calibration_budget,
            self.config.calibration_strata,
            _calibration_rng(),
        )
        if not picks:
            return math.inf, -math.inf
        calibration_candidates = [usable[pick][0] for pick in picks]
        pick_of = {usable[pick][0].candidate_id: pick for pick in picks}
        labels: dict[int, int] = {}
        calibration_batches = list(
            _blocks(calibration_candidates, self.config.expensive_batch_size)
        )
        calibration_started = time.perf_counter()
        for batch_index, batch in enumerate(calibration_batches, 1):
            metrics.calibration_candidates += len(batch)
            oracle_accepts, decided = self._expensive_decide(
                batch, predicate, metrics, calibration=True
            )
            _print_progress(
                "Batch-wise cascade threshold calibration",
                batch_index,
                len(calibration_batches),
                calibration_started,
            )
            metrics.calibration_expensive_accepts += len(oracle_accepts)
            for candidate in batch:
                if candidate.candidate_id not in decided:
                    continue
                accepted = candidate.candidate_id in oracle_accepts
                labels[pick_of[candidate.candidate_id]] = 1 if accepted else 0
                if calibration_oracle is not None:
                    calibration_oracle[candidate.candidate_id] = accepted

        if not labels:
            metrics.calibration_agreement = 0.0
            metrics.calibration_activated = False
            return math.inf, -math.inf

        fit = fit_band(
            pool_scores,
            labels,
            quality_target=self.config.cascade_target,
            credible_level=self.config.credible_level,
            n_strata=self.config.calibration_strata,
        )
        metrics.calibration_accept_precision_lower = fit.accept_precision_lower
        metrics.calibration_reject_recall_lower = fit.reject_recall_lower

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
        metrics.calibration_agreement = agreed / selected_count if selected_count else 0.0
        metrics.calibration_activated = fit.activated
        return fit.accept_threshold, fit.reject_threshold


def exact_id_candidates(
    movies: Iterable[dict[str, str]],
    reviews: Iterable[dict[str, str]],
) -> list[Candidate]:
    movies_by_id: dict[str, list[dict[str, str]]] = {}
    for movie in movies:
        movies_by_id.setdefault(
            str(movie.get("movie_id", "")),
            [],
        ).append(movie)
    candidates: list[Candidate] = []
    for review in reviews:
        for movie in movies_by_id.get(str(review.get("tconst", "")), []):
            candidates.append(
                Candidate(len(candidates) + 1, movie, review)
            )
    return candidates


def metrics_dict(metrics: CascadeMetrics) -> dict[str, object]:
    return asdict(metrics)


def _print_progress(label: str, done: int, total: int, started: float) -> None:
    if total <= 0:
        return
    elapsed = time.perf_counter() - started
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    percent = 100.0 * done / total
    width = 24
    filled = min(width, int(width * done / total))
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"{label}: [{bar}] {done}/{total} ({percent:.1f}%) "
        f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
        flush=True,
    )


def joined_row(candidate: Candidate, source: str) -> dict[str, str]:
    movie, review = candidate.movie, candidate.review
    return {
        "candidate_id": str(candidate.candidate_id),
        "movie_id": movie.get("movie_id", ""),
        "title": movie.get("title", ""),
        "year": movie.get("year", ""),
        "director": movie.get("director", ""),
        "runtime": movie.get("runtime", ""),
        "genres": movie.get("genres", ""),
        "tconst": review.get("tconst", ""),
        "review": review.get("review", ""),
        "match_source": source,
    }


def _set_time_percentages(metrics: CascadeMetrics) -> None:
    model_seconds = metrics.cheap_seconds + metrics.expensive_seconds
    if model_seconds <= 0:
        return
    metrics.cheap_time_percent = 100.0 * metrics.cheap_seconds / model_seconds
    metrics.expensive_time_percent = (
        100.0 * metrics.expensive_seconds / model_seconds
    )


def _decision(
    candidate: Candidate,
    score: float | None,
    route: str,
    error: str = "",
) -> Decision:
    return Decision(
        candidate.candidate_id,
        candidate.movie.get("movie_id", ""),
        candidate.review.get("tconst", ""),
        score,
        route,
        error,
    )


def _cheap_batch_prompt(
    candidates: list[Candidate],
    predicate: str,
    max_review_chars: int,
) -> str:
    parts = [
        "Classify every explicit candidate pair below.",
        f"Predicate: {predicate}",
        semantic_guideline(predicate),
        CHEAP_EVIDENCE_INSTRUCTIONS,
        "Return one line per candidate as CANDIDATE_ID: YES, NO, or UNCERTAIN.",
        "Do not include confidence, evidence, explanations, or prose.",
        "Return every candidate exactly once and do not explain.",
        "",
    ]
    for candidate in candidates:
        parts.extend(
            [
                f"CANDIDATE_{candidate.candidate_id}:",
                f"Movie: {candidate.movie.get('text', '')}",
                f"Review: {candidate.review.get('text', '')[:max_review_chars]}",
                "",
            ]
        )
    parts.append("Decisions:")
    return "\n".join(parts)


def _expensive_batch_prompt(
    candidates: list[Candidate],
    predicate: str,
    max_review_chars: int,
) -> str:
    parts = [
        "Evaluate the explicit candidate pairs below.",
        f"Predicate: {predicate}",
        semantic_guideline(predicate),
        EXPENSIVE_EVIDENCE_INSTRUCTIONS,
        "Return one line per candidate as PAIR_ID: YES or PAIR_ID: NO.",
        "Do not include confidence, evidence, explanations, or prose.",
        "Return every candidate exactly once. Do not add prose, JSON, or markdown.",
        "",
    ]
    for candidate in candidates:
        parts.extend(
            [
                f"PAIR_{candidate.candidate_id}:",
                f"Movie: {candidate.movie.get('text', '')}",
                f"Review: {candidate.review.get('text', '')[:max_review_chars]}",
                "",
            ]
        )
    parts.append("Decisions:")
    return "\n".join(parts)


def _parse_batch_scores(
    payload: dict,
    valid: set[int],
) -> dict[int, float]:
    answer = str(payload.get("message", {}).get("content", ""))
    plain = _plain_decisions(answer, valid)
    if set(plain) == valid:
        return {
            candidate_id: 2.0 if label == "yes" else (0.0 if label == "uncertain" else -2.0)
            for candidate_id, label in plain.items()
        }
    parsed = _extract_json(answer)
    if parsed is None:
        raise ValueError(
            "cheap batch response omitted candidate decisions: " + answer[:200]
        )
    scores: dict[int, float] = {}
    decisions: object
    if isinstance(parsed, dict):
        decisions = parsed.get(
            "decisions",
            parsed.get("answers", parsed.get("results", parsed)),
        )
    else:
        decisions = parsed
    if isinstance(decisions, dict):
        decisions = [
            {"candidate_id": candidate_id, "answer": answer_value}
            for candidate_id, answer_value in decisions.items()
        ]
    if not isinstance(decisions, list):
        decisions = []
    for item in decisions:
        if not isinstance(item, dict):
            continue
        try:
            candidate_id = int(
                item.get("candidate_id", item.get("pair_id", item.get("id")))
            )
        except (KeyError, TypeError, ValueError):
            continue
        label = _binary_answer(
            item.get(
                "answer",
                item.get("label", item.get("decision", item.get("match"))),
            )
        )
        if candidate_id in valid and label is not None:
            scores[candidate_id] = 2.0 if label == "1" else (0.0 if label == "u" else -2.0)
    if set(scores) != valid:
        missing = sorted(valid - set(scores))
        raise ValueError(f"cheap batch response omitted candidate IDs: {missing}")
    return scores


def _parse_matching_ids(
    answer: str,
    candidates: list[Candidate],
    valid: set[int],
) -> set[int]:
    accepted: set[int] = set()
    plain = _plain_decisions(answer, valid)
    accepted.update(
        candidate_id
        for candidate_id, label in plain.items()
        if label in {"yes", "uncertain"}
    )
    structured = _extract_json(answer)
    if isinstance(structured, dict):
        decisions = structured.get("decisions", [])
        if isinstance(decisions, list):
            for item in decisions:
                if not isinstance(item, dict):
                    continue
                try:
                    candidate_id = int(item.get("candidate_id"))
                except (TypeError, ValueError):
                    continue
                if (
                    candidate_id in valid
                    and str(item.get("decision", "")).strip().lower() == "yes"
                ):
                    accepted.add(candidate_id)
        for key in (
            "matching_pair_ids",
            "matching_candidate_ids",
            "pair_ids",
            "matches",
        ):
            values = structured.get(key, [])
            if isinstance(values, list):
                for value in values:
                    try:
                        candidate_id = int(value)
                    except (TypeError, ValueError):
                        continue
                    if candidate_id in valid:
                        accepted.add(candidate_id)
    if not plain:
        accepted.update(
            int(value)
            for value in PAIR_LABEL_RE.findall(answer)
            if int(value) in valid
        )
    for token in answer.split(","):
        match = PLAIN_NUMBER_RE.match(token)
        if match and int(match.group(1)) in valid:
            accepted.add(int(match.group(1)))
    by_movie_id: dict[str, list[int]] = {}
    for candidate in candidates:
        by_movie_id.setdefault(
            str(candidate.movie.get("movie_id", "")),
            [],
        ).append(candidate.candidate_id)
    for movie_id, candidate_ids in by_movie_id.items():
        if movie_id and len(candidate_ids) == 1 and movie_id in answer:
            accepted.add(candidate_ids[0])
    return accepted


def _parse_expensive_decisions(answer: str, valid: set[int]) -> dict[int, str]:
    """Parse complete positive and negative decisions from any supported form."""
    decisions = _plain_decisions(answer, valid)
    structured = _extract_json(answer)
    if isinstance(structured, dict) and isinstance(structured.get("decisions"), list):
        for item in structured["decisions"]:
            if not isinstance(item, dict):
                continue
            try:
                candidate_id = int(item.get("candidate_id", item.get("pair_id")))
            except (TypeError, ValueError):
                continue
            label = _binary_answer(item.get("decision", item.get("answer")))
            if candidate_id in valid and label is not None:
                decisions[candidate_id] = {"1": "yes", "0": "no", "u": "uncertain"}[label]
    return {
        candidate_id: label
        for candidate_id, label in decisions.items()
        if candidate_id in valid and label in {"yes", "no"}
    }


def _plain_decisions(answer: str, valid: set[int]) -> dict[int, str]:
    """Read tolerant ``PAIR_7: YES`` / ``7, no`` plain-text decisions."""
    decisions: dict[int, str] = {}
    pattern = re.compile(
        r"(?:candidate|pair)?[_ #:-]*(\d+)\s*(?:[:=,|\-]|\s)\s*"
        r"(yes|no|uncertain|true|false|match|matched|not matched)\b",
        re.IGNORECASE,
    )
    for raw_id, raw_label in pattern.findall(answer):
        candidate_id = int(raw_id)
        label = str(raw_label).strip().lower()
        if candidate_id not in valid:
            continue
        if label in {"true", "match", "matched"}:
            label = "yes"
        elif label in {"false", "not matched"}:
            label = "no"
        decisions[candidate_id] = label
    return decisions


def _binary_answer(value: object) -> str | None:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)) and value in (0, 1):
        return str(int(value))
    text = str(value).strip().lower()
    if text in {"1", "yes", "true", "match", "matched"}:
        return "1"
    if text in {"0", "no", "false", "no match", "not matched"}:
        return "0"
    if text in {"uncertain", "unknown", "maybe"}:
        return "u"
    return None


def _extract_json(text: str) -> object | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def _is_valid_decision_answer(answer: str) -> bool:
    parsed = _extract_json(answer)
    if isinstance(parsed, dict):
        decisions = parsed.get("decisions")
        if isinstance(decisions, list) and decisions:
            for item in decisions:
                if not isinstance(item, dict):
                    return False
                try:
                    int(item["candidate_id"])
                except (KeyError, TypeError, ValueError):
                    return False
                if (
                    str(item.get("decision", "")).strip().lower()
                    not in {"yes", "no", "uncertain"}
                ):
                    return False
            return True
        return any(
            key in parsed
            for key in (
                "matching_pair_ids",
                "matching_candidate_ids",
                "pair_ids",
                "matches",
            )
        )
    if parsed == []:
        return True
    return bool(
        re.search(
            r"\b(?:none|no matches|no matching (?:pairs|candidates))\b",
            answer,
            re.IGNORECASE,
        )
    )


def _has_complete_decision_coverage(answer: str, valid: set[int]) -> bool:
    parsed = _extract_json(answer)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("decisions"), list):
        return False
    returned: list[int] = []
    for item in parsed["decisions"]:
        if not isinstance(item, dict):
            return False
        try:
            candidate_id = int(item["candidate_id"])
        except (KeyError, TypeError, ValueError):
            return False
        if str(item.get("decision", "")).strip().lower() not in {"yes", "no", "uncertain"}:
            return False
        returned.append(candidate_id)
    return len(returned) == len(valid) and set(returned) == valid


def _plain_model(model: str) -> str:
    return model.removeprefix("ollama/")


def _calibration_rng() -> "random.Random | None":
    """Per-repetition RNG for the calibration draw, or None when disabled.

    Every LLM call runs at temperature 0, so the calibration sample is the only
    step that can legitimately differ between repetitions.  CALIBRATION_SEED is
    set per repetition by benchmarks/shared/scripts/repetitions.py; unset (or
    'none') keeps the fully deterministic evenly-spaced pick.
    """
    raw = os.environ.get("CALIBRATION_SEED", "").strip()
    if not raw or raw.lower() == "none":
        return None
    return random.Random(raw)


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _blocks(rows: list[Candidate], size: int) -> Iterable[list[Candidate]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _deduplicate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    result = []
    for row in rows:
        key = (row["movie_id"], row["tconst"], row["review"])
        if key not in seen:
            seen.add(key)
            result.append(row)
    return result
