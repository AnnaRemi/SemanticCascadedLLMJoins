from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from .client import ChatClient
from .semantic_dict_context import semantic_guideline

try:
    import tiktoken
except ImportError:  # pragma: no cover
    tiktoken = None


PAIR_RE = re.compile(r"(\d+)\s*,\s*(\d+)")


@dataclass
class JoinStats:
    block_1: int
    block_2: int
    tokens_read: int
    tokens_written: int
    seconds: float
    overflow: bool
    prompt_tokens_estimated: int
    parsed_pairs: int
    raw_response: str


def token_size(text: str, model: str = "gpt-4o") -> int:
    """Return token size, using tiktoken when available.

    The official implementation uses `tiktoken.encoding_for_model('gpt-4o')`.
    This keeps that behavior but falls back to a conservative word/punctuation
    estimate so dry runs still work without optional dependencies.
    """

    if tiktoken is not None:
        try:
            encoder = tiktoken.encoding_for_model(model.removeprefix("ollama/"))
        except Exception:
            encoder = tiktoken.get_encoding("cl100k_base")
        return len(encoder.encode(text))
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))


def tuple_size(df: pd.DataFrame, token_model: str = "gpt-4o") -> float:
    if df.empty:
        return 1.0
    return float(df["text"].map(lambda value: token_size(str(value), token_model)).mean())


def optimal_block_size(
    s1: float,
    s2: float,
    s3: float,
    token_threshold: int,
    prompt_size: int,
    selectivity_estimate: float,
) -> tuple[int, int]:
    """Port of `llmjoin.common.tuning.optimal_block_size` with guards."""

    estimate = max(float(selectivity_estimate), 0.0000001)
    available = max(1.0, float(token_threshold - prompt_size))
    numerator = math.sqrt(s1 * s1 * s2 * s2 + s1 * s2 * s3 * estimate * available) - s1 * s2
    b1 = math.floor(numerator / (s1 * s3 * estimate))
    b1 = max(1, b1)
    b2 = math.floor(available - b1 * s1)
    b2 = math.floor(b2 / max(1.0, s2 + b1 * s3 * estimate))
    b2 = max(1, b2)
    return b1, b2


def create_prompt(
    block_1: list[dict],
    block_2: list[dict],
    predicate: str,
) -> str:
    """Create the Trummer-style block join prompt for product/review rows."""

    parts = [
        "Find every product in Collection 1 that has a review in Collection 2 "
        "satisfying this predicate:",
        predicate,
        semantic_guideline(predicate),
        "Act as a recall-first semantic filter, while requiring review-specific evidence.",
        "- Answer YES for direct wording, synonyms/paraphrases, described examples, or reasonable implications.",
        "- Give borderline evidence the benefit of the doubt when it specifically bears on the predicate.",
        "- Count explicit mentions even when negated, qualified, quoted, or critical; retrieve evidence rather than score sentiment.",
        "- Answer NO when support is absent, generic, based only on genre/topic, or never discusses the predicate itself.",
        "- Lack of contradiction alone is not evidence.",
        "Collection 1 contains structured product rows. Collection 2 contains review chunks.",
        "A review belongs to a product only when its product_id is exactly equal to the product product_id.",
        "Return one decision per line as: product_id: YES or product_id: NO.",
        "Do not include confidence, evidence, explanations, or prose.",
        "Copy each product_id exactly from Collection 1.",
        "Do not add prose, JSON, markdown, or a completion marker.",
        "",
        "Collection 1:",
    ]
    for idx, row in enumerate(block_1, 1):
        parts.append(f"{idx}: {row['text']}")
    parts.append("")
    parts.append("Collection 2:")
    for idx, row in enumerate(block_2, 1):
        parts.append(f"{idx}: {row['text']}")
    parts.append("")
    parts.append("Decisions:")
    return "\n".join(parts)


def partition_records(df: pd.DataFrame, block_size: int) -> list[list[dict]]:
    records = df.to_dict("records")
    return [records[i : i + block_size] for i in range(0, len(records), block_size)]


def print_progress(label: str, done: int, total: int, started: float) -> None:
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


def parse_pairs(answer: str, block_1: list[dict], block_2: list[dict]) -> list[dict]:
    results: list[dict] = []
    seen: set[tuple[str, str]] = set()
    products_by_id = {
        str(product.get("product_id", "")): product
        for product in block_1
    }
    reviews_by_id: dict[str, list[dict]] = {}
    for review in block_2:
        reviews_by_id.setdefault(str(review.get("product_id", "")), []).append(review)
    id_pairs: list[tuple[str, str]] = []
    index_pairs: list[tuple[object, object]] = []
    payload = _extract_json(answer)
    if isinstance(payload, dict):
        try:
            for decision in payload.get("decisions", []):
                if not isinstance(decision, dict):
                    continue
                product_id = str(decision.get("product_id", ""))
                label = str(decision.get("decision", "")).strip().lower()
                if (
                    product_id in products_by_id
                    and label == "yes"
                ):
                    for review in reviews_by_id.get(product_id, []):
                        _append_result(results, seen, products_by_id[product_id], review)
            for product_id in payload.get(
                "matching_product_ids",
                payload.get("product_ids", payload.get("matches", [])),
            ):
                product_id = str(product_id)
                if product_id not in products_by_id:
                    continue
                for review in reviews_by_id.get(product_id, []):
                    _append_result(
                        results, seen, products_by_id[product_id], review
                    )
            for pair in payload.get("pairs", []):
                if isinstance(pair, dict):
                    if "product_id" in pair and "product_id" in pair:
                        id_pairs.append(
                            (str(pair.get("product_id", "")), str(pair.get("product_id", "")))
                        )
                    elif "product_index" in pair and "review_index" in pair:
                        index_pairs.append(
                            (pair.get("product_index"), pair.get("review_index"))
                        )
                elif isinstance(pair, list) and len(pair) == 2:
                    index_pairs.append((pair[0], pair[1]))
        except (TypeError, ValueError):
            pass

    # Preferred schema-free protocol: one ``product_id: decision`` per line.
    # Amazon ASINs (e.g. "B00008JVTT") don't match a fixed shape the way
    # IMDb's "tt\d+" tconst does, so match against the known product ids
    # from this block instead of a hardcoded id pattern.
    if products_by_id:
        ids_alternation = "|".join(re.escape(pid) for pid in products_by_id if pid)
        for product_id, label in re.findall(
            rf"\b({ids_alternation})\b\s*(?:[:=,|\-]|\s)\s*(?i:(yes|no|uncertain))\b",
            answer,
        ):
            if label.lower() != "yes" or product_id not in products_by_id:
                continue
            for review in reviews_by_id.get(product_id, []):
                _append_result(results, seen, products_by_id[product_id], review)

    for product_id, review_key in id_pairs:
        if product_id != review_key or product_id not in products_by_id:
            continue
        for review in reviews_by_id.get(review_key, []):
            _append_result(results, seen, products_by_id[product_id], review)

    if not id_pairs and not index_pairs:
        index_pairs = PAIR_RE.findall(answer)
    for raw_x, raw_y in index_pairs:
        try:
            idx_1 = int(raw_x) - 1
            idx_2 = int(raw_y) - 1
        except (TypeError, ValueError):
            continue
        if not (0 <= idx_1 < len(block_1) and 0 <= idx_2 < len(block_2)):
            continue
        product = block_1[idx_1]
        review = block_2[idx_2]
        if str(product.get("product_id", "")) != str(review.get("product_id", "")):
            continue
        _append_result(results, seen, product, review)
    return results


def parse_decisions(answer: str, block_1: list[dict]) -> dict[str, str]:
    """Return every explicit product decision found in JSON or plain text."""
    valid = {str(product.get("product_id", "")) for product in block_1}
    decisions: dict[str, str] = {}
    payload = _extract_json(answer)
    if isinstance(payload, dict) and isinstance(payload.get("decisions"), list):
        for item in payload["decisions"]:
            if not isinstance(item, dict):
                continue
            product_id = str(item.get("product_id", ""))
            label = str(item.get("decision", "")).strip().lower()
            if product_id in valid and label in {"yes", "no", "uncertain"}:
                decisions[product_id] = label
    if valid:
        ids_alternation = "|".join(re.escape(pid) for pid in valid if pid)
        for product_id, label in re.findall(
            rf"\b({ids_alternation})\b\s*(?:[:=,|\-]|\s)\s*(?i:(yes|no|uncertain))\b",
            answer,
        ):
            if product_id in valid:
                decisions[product_id] = label.lower()
    return decisions


def has_complete_decision_coverage(answer: str, block_1: list[dict]) -> bool:
    payload = _extract_json(answer)
    if not isinstance(payload, dict) or not isinstance(payload.get("decisions"), list):
        return False
    expected = [str(product.get("product_id", "")) for product in block_1]
    returned: list[str] = []
    for item in payload["decisions"]:
        if not isinstance(item, dict):
            return False
        product_id = str(item.get("product_id", ""))
        decision = str(item.get("decision", "")).strip().lower()
        if product_id not in expected or decision not in {"yes", "no", "uncertain"}:
            return False
        returned.append(product_id)
    return len(returned) == len(expected) and set(returned) == set(expected)


def _extract_json(answer: str) -> object | None:
    stripped = answer.strip()
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


def _append_result(
    results: list[dict],
    seen: set[tuple[str, str]],
    product: dict,
    review: dict,
) -> None:
    key = (str(product.get("product_id", "")), str(review.get("product_id", "")))
    if key in seen:
        return
    seen.add(key)
    results.append(
        {
            "product_id": product.get("product_id", ""),
            "title": product.get("title", ""),
            "brand": product.get("brand", ""),
            "category": product.get("category", ""),
            "price": product.get("price", ""),
            "product_id": review.get("product_id", ""),
            "review": review.get("review", ""),
        }
    )


def join_two_blocks(
    client: ChatClient,
    block_1: list[dict],
    block_2: list[dict],
    predicate: str,
    model: str,
    token_threshold: int,
    token_model: str,
    block_1_index: int,
    block_2_index: int,
    max_completion_tokens: int | None = None,
    dry_run: bool = False,
) -> tuple[JoinStats, list[dict]]:
    start_s = time.time()
    review_ids = {str(review.get("product_id", "")) for review in block_2}
    active_block_1 = [
        product
        for product in block_1
        if str(product.get("product_id", "")) in review_ids
    ]
    if not active_block_1:
        return (
            JoinStats(
                block_1_index, block_2_index, 0, 0,
                time.time() - start_s, False, 0, 0, ""
            ),
            [],
        )
    prompt = create_prompt(active_block_1, block_2, predicate)
    prompt_tokens = token_size(prompt, token_model)
    max_tokens = token_threshold - prompt_tokens
    if max_completion_tokens is not None:
        max_tokens = min(max_tokens, max_completion_tokens)
    # Plain-text decisions are short. Bound generation by the number of
    # candidates so a verbose model cannot spend hundreds of tokens after the
    # useful answer has already been produced.
    max_tokens = min(max_tokens, max(48, 16 + 16 * len(active_block_1)))
    if max_tokens < 1:
        return (
            JoinStats(
                block_1_index, block_2_index, 0, 0,
                time.time() - start_s, True, prompt_tokens, 0, ""
            ),
            [],
        )

    if dry_run:
        results = _deterministic_dry_run(active_block_1, block_2)
        return (
            JoinStats(
                block_1_index, block_2_index, prompt_tokens, 0,
                time.time() - start_s, False, prompt_tokens, len(results), ""
            ),
            results,
        )

    try:
        response = client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=max_tokens,
            temperature=0,
        )
    except Exception as exc:
        return (
            JoinStats(
                block_1_index,
                block_2_index,
                0,
                0,
                time.time() - start_s,
                True,
                prompt_tokens,
                0,
                f"request_failed: {type(exc).__name__}: {exc}",
            ),
            [],
        )
    decisions = parse_decisions(response.content, active_block_1)
    results = parse_pairs(response.content, active_block_1, block_2)
    overflow = response.finish_reason == "length" or len(decisions) < len(active_block_1)
    return (
        JoinStats(
            block_1_index,
            block_2_index,
            response.prompt_tokens,
            response.completion_tokens,
            time.time() - start_s,
            overflow,
            prompt_tokens,
            len(results),
            response.content,
        ),
        results,
    )


def block_join(
    client: ChatClient,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    predicate: str,
    model: str,
    selectivity_estimate: float = 0.01,
    token_threshold: int = 4000,
    token_model: str = "gpt-4o",
    max_completion_tokens: int | None = None,
    max_block_1_size: int | None = None,
    max_block_2_size: int | None = None,
    dry_run: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    s1 = tuple_size(df1, token_model)
    s2 = tuple_size(df2, token_model)
    result_tuple_size = 8.0
    static_prompt_size = token_size(create_prompt([], [], predicate), token_model)
    b1, b2 = optimal_block_size(
        s1,
        s2,
        result_tuple_size,
        token_threshold,
        static_prompt_size,
        selectivity_estimate,
    )
    if max_block_1_size is not None:
        b1 = min(b1, max(1, max_block_1_size))
    if max_block_2_size is not None:
        b2 = min(b2, max(1, max_block_2_size))

    blocks_1 = partition_records(df1, b1)
    blocks_2 = partition_records(df2, b2)

    all_stats: list[JoinStats] = []
    all_results: list[dict] = []
    overflow = False
    total_blocks = len(blocks_1) * len(blocks_2)
    completed_blocks = 0
    progress_started = time.perf_counter()
    for idx_1, block_1 in enumerate(blocks_1, 1):
        if overflow:
            break
        for idx_2, block_2 in enumerate(blocks_2, 1):
            print(
                f"Joining block {completed_blocks + 1}/{total_blocks}: "
                f"{idx_1}/{len(blocks_1)} with "
                f"{idx_2}/{len(blocks_2)} "
                f"({len(block_1)}x{len(block_2)} rows)",
                flush=True,
            )
            stats, results = join_two_blocks(
                client,
                block_1,
                block_2,
                predicate,
                model,
                token_threshold,
                token_model,
                idx_1,
                idx_2,
                max_completion_tokens=max_completion_tokens,
                dry_run=dry_run,
            )
            all_stats.append(stats)
            all_results.extend(results)
            overflow = stats.overflow
            completed_blocks += 1
            print_progress("Block join progress", completed_blocks, total_blocks, progress_started)
            if overflow:
                break

    return pd.DataFrame([s.__dict__ for s in all_stats]), pd.DataFrame(all_results)


def adaptive_join(
    client: ChatClient,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    predicate: str,
    model: str,
    initial_selectivity: float = 0.001,
    token_threshold: int = 4000,
    token_model: str = "gpt-4o",
    max_completion_tokens: int | None = None,
    max_block_1_size: int | None = None,
    max_block_2_size: int | None = None,
    dry_run: bool = False,
    max_rounds: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    estimate = initial_selectivity
    stats_frames: list[pd.DataFrame] = []
    result = pd.DataFrame()

    for round_nr in range(1, max_rounds + 1):
        stats, result = block_join(
            client,
            df1,
            df2,
            predicate,
            model,
            selectivity_estimate=estimate,
            token_threshold=token_threshold,
            token_model=token_model,
            max_completion_tokens=max_completion_tokens,
            max_block_1_size=max_block_1_size,
            max_block_2_size=max_block_2_size,
            dry_run=dry_run,
        )
        if not stats.empty:
            stats["round"] = round_nr
            stats["selectivity_estimate"] = estimate
        stats_frames.append(stats)
        if stats.empty or not bool(stats["overflow"].any()):
            break
        estimate *= 4

    all_stats = pd.concat(stats_frames, ignore_index=True) if stats_frames else pd.DataFrame()
    if "nonempty_fallback_rows" not in all_stats:
        all_stats["nonempty_fallback_rows"] = 0
    return all_stats, result



def _batch_texts(batch: list[dict], texts_by_id: dict[str, list[dict]], id_1: str) -> list[dict]:
    return [t for row in batch for t in texts_by_id.get(str(row.get(id_1, "")), [])]


def _fit_batches(
    rows: list[dict],
    texts_by_id: dict[str, list[dict]],
    id_1: str,
    predicate: str,
    token_model: str,
    token_threshold: int,
    max_completion_tokens: int | None,
    max_rows: int,
) -> list[list[dict]]:
    """Pack rows into batches that fit the token budget.

    join_two_blocks refuses to call the model when the prompt leaves no room for
    a completion, and a refused block yields no rows at all.  Sizing batches
    purely by row count therefore silently drops every batch whose prompt is too
    large, so the budget -- not just ``max_rows`` -- decides where batches end.
    """
    batches: list[list[dict]] = []
    current: list[dict] = []
    for row in rows:
        trial = current + [row]
        prompt = create_prompt(trial, _batch_texts(trial, texts_by_id, id_1), predicate)
        prompt_tokens = token_size(prompt, token_model)
        wanted = max(48, 16 + 16 * len(trial))
        if max_completion_tokens is not None:
            wanted = min(wanted, max_completion_tokens)
        too_big = token_threshold - prompt_tokens < wanted
        if current and (too_big or len(trial) > max_rows):
            batches.append(current)
            current = [row]
        else:
            current = trial
    if current:
        batches.append(current)
    return batches


def single_pass_join(
    client: ChatClient,
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    predicate: str,
    model: str,
    block_size: int = 25,
    token_threshold: int = 4000,
    token_model: str = "gpt-4o",
    max_completion_tokens: int | None = None,
    dry_run: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Judge every row exactly once, in batches: ceil(n / block_size) calls.

    The adaptive block join this replaces issued one call per (row-block x
    text-block) pair and repeated the whole sweep for up to ``max_rounds``
    rounds, so its cost was rounds x b1 x b2 rather than n/b.  Here each row is
    paired with its own text up front, the pairs are cut into batches, and each
    batch costs exactly one expensive call.
    """
    id_1 = "product_id" if "product_id" in df1.columns else "movie_id"
    id_2 = "product_id" if "product_id" in df2.columns else "tconst"

    texts_by_id: dict[str, list[dict]] = {}
    for record in df2.to_dict("records"):
        texts_by_id.setdefault(str(record.get(id_2, "")), []).append(record)

    rows = df1.to_dict("records")
    batches = _fit_batches(
        rows,
        texts_by_id,
        id_1,
        predicate,
        token_model,
        token_threshold,
        max_completion_tokens,
        block_size,
    )

    all_stats: list[JoinStats] = []
    all_results: list[dict] = []
    started = time.perf_counter()
    for index, batch in enumerate(batches, 1):
        paired_texts = _batch_texts(batch, texts_by_id, id_1)
        if not paired_texts:
            continue
        print(
            f"Joining batch {index}/{len(batches)} "
            f"({len(batch)} rows, {len(paired_texts)} texts)",
            flush=True,
        )
        stats, results = join_two_blocks(
            client,
            batch,
            paired_texts,
            predicate,
            model,
            token_threshold,
            token_model,
            index,
            1,
            max_completion_tokens=max_completion_tokens,
            dry_run=dry_run,
        )
        all_stats.append(stats)
        all_results.extend(results)
        print_progress("batches", index, len(batches), started)

    stats_frame = pd.DataFrame([s.__dict__ for s in all_stats]) if all_stats else pd.DataFrame()
    if "nonempty_fallback_rows" not in stats_frame:
        stats_frame["nonempty_fallback_rows"] = 0
    result_frame = pd.DataFrame(all_results) if all_results else pd.DataFrame()
    return stats_frame, result_frame

def _deterministic_dry_run(block_1: Iterable[dict], block_2: Iterable[dict]) -> list[dict]:
    """Cheap local check used only for smoke tests.

    It verifies the row wiring by matching keys and simple visual-effects terms.
    Real semantic matching is done by the LLM path.
    """

    terms = ("visual", "effect", "effects", "cgi", "innovative", "stunning", "spectacle")
    results: list[dict] = []
    for product in block_1:
        product_id = str(product.get("product_id", ""))
        for review in block_2:
            text = str(review.get("review", "")).lower()
            if product_id == str(review.get("product_id", "")) and any(term in text for term in terms):
                results.append(
                    {
                        "product_id": product.get("product_id", ""),
                        "title": product.get("title", ""),
                        "brand": product.get("brand", ""),
                        "category": product.get("category", ""),
                        "price": product.get("price", ""),
                        "product_id": review.get("product_id", ""),
                        "review": review.get("review", ""),
                    }
                )
    return results
