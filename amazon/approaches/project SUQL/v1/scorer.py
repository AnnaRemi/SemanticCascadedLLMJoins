from __future__ import annotations

import math
import os
from typing import Iterable

import httpx

from semantic_dict_context import semantic_guideline


DEFAULT_MODEL = os.environ.get("SUQL_MODEL", "ollama/gemma4:e2b")
DEFAULT_API_BASE = os.environ.get("SUQL_API_BASE", "http://localhost:11434")


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


def _logsumexp(values: list[float]) -> float:
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values))


def native_log_odds(payload: dict) -> float | None:
    """Real log-odds from Ollama's native ``logprobs`` (``/api/generate`` with
    ``logprobs: true, top_logprobs: N``), or None when the server returned none.

    Every yes-like token (1, Yes, yes, ...) and every no-like token (0, No, ...)
    among the top alternatives of the first generated token is pooled, so the
    score is log P(yes) - log P(no) rather than the difference of two arbitrary
    spellings. If only one side appears in the top-N, the other side is treated
    as the remaining probability mass (same convention as ``_single_score``).
    """
    entries = payload.get("logprobs")
    if not isinstance(entries, list) or not entries or not isinstance(entries[0], dict):
        return None
    first = entries[0]
    alternatives = first.get("top_logprobs") or [first]
    yes: list[float] = []
    no: list[float] = []
    for item in alternatives:
        if not isinstance(item, dict) or item.get("logprob") is None:
            continue
        label = _label(str(item.get("token", "")))
        if label == "1":
            yes.append(float(item["logprob"]))
        elif label == "0":
            no.append(float(item["logprob"]))
    if yes and no:
        return _logsumexp(yes) - _logsumexp(no)
    if yes:
        return _single_score("1", _logsumexp(yes))
    if no:
        return _single_score("0", _logsumexp(no))
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


class OllamaLogOddsScorer:
    def __init__(self, model: str | None = None, api_base: str | None = None, timeout: float = 120.0) -> None:
        self.model = model or DEFAULT_MODEL
        self.ollama_model = self.model.removeprefix("ollama/")
        self.api_base = (api_base or DEFAULT_API_BASE).rstrip("/")
        self.timeout = timeout
        self._openai_completions_available: bool | None = None
        self._native_logprobs_available: bool | None = None
        # How the last score was obtained: "native_logprobs" (real log-odds),
        # "v1_completions" or "label_only" (+-2 yes/no, no confidence at all).
        self.score_mode = ""

    def score(self, review: str, question: str) -> float:
        prompt = self._prompt(review, question)
        if self._native_logprobs_available is not False:
            try:
                response = httpx.post(
                    f"{self.api_base}/api/generate",
                    json={"model": self.ollama_model, "prompt": prompt, "stream": False,
                          "think": False, "logprobs": True, "top_logprobs": 10,
                          "options": {"temperature": 0, "num_predict": 1}},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                score = native_log_odds(response.json())
                if score is not None:
                    self._native_logprobs_available = True
                    self.score_mode = "native_logprobs"
                    return score
                # Request accepted but no log-probs came back (older Ollama).
                self._native_logprobs_available = False
            except Exception:
                pass  # transient failure: fall through to the legacy path for this row
        openai_error: Exception | None = None
        if self._openai_completions_available is not False:
            try:
                response = httpx.post(
                    f"{self.api_base}/v1/completions",
                    json={"model": self.ollama_model, "prompt": prompt, "max_tokens": 1,
                          "temperature": 0, "logprobs": 10, "think": False},
                    timeout=self.timeout,
                )
                if response.status_code == 404:
                    self._openai_completions_available = False
                else:
                    response.raise_for_status()
                    score = extract_binary_log_odds(response.json())
                    self._openai_completions_available = True
                    self.score_mode = "v1_completions" if abs(score) != 2.0 else "label_only"
                    return score
            except Exception as exc:
                openai_error = exc
                self._openai_completions_available = False
        try:
            response = httpx.post(
                f"{self.api_base}/api/generate",
                json={"model": self.ollama_model, "prompt": prompt, "stream": False,
                      "think": False, "options": {"temperature": 0, "num_predict": 1}},
                timeout=self.timeout,
            )
            response.raise_for_status()
            score = extract_binary_log_odds(response.json())
            self.score_mode = "label_only"
            return score
        except Exception as native_error:
            if openai_error is not None:
                raise RuntimeError(
                    f"Both scoring APIs failed: {openai_error}; {native_error}"
                ) from native_error
            raise

    @staticmethod
    def _prompt(review: str, question: str) -> str:
        guidance = semantic_guideline(question)
        guidance_block = f"{guidance}\n\n" if guidance else ""
        return (
            "Act as a high-recall first-pass semantic filter. Decide whether this review "
            "contains review-specific evidence for a Yes answer. Count direct wording, "
            "synonyms, described examples, and reasonable implications as evidence. Give "
            "borderline but condition-specific evidence the benefit of the doubt. Do not "
            "discard an explicit mention because it is negated, qualified, quoted, or "
            "critical: this is evidence retrieval, not sentiment scoring. Do not "
            "count genre/topic alone, generic praise or criticism, or mere lack of "
            "contradiction as evidence.\n"
            "Return exactly one token: 1 for Yes, 0 for No.\n\n"
            + guidance_block
            + f"Question: {question}\nReview: {review[:1800]}\n\nAnswer:"
        )


def parse_label(value: object) -> int:
    label = _label(str(value))
    if label is None:
        raise ValueError(f"Unsupported label value: {value!r}")
    return int(label)
