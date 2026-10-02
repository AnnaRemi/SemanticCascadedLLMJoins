"""
suql_engine.py
==============
Core SUQL engine for the IMDb movie database.

SUQL (Structured and Unstructured Query Language) — Stanford OVAL paper:
  https://arxiv.org/abs/2311.09818

Architecture (mirrors the paper):
  1. NL question  →  SUQL query  (LLM semantic parser, few-shot in-context learning)
  2. SUQL query   →  execute structured SQL part on pandas DataFrame
  3. For each row that passes structured filters, evaluate  answer(review, question)
     by calling an LLM — returns 'Yes' / 'No' / a short answer string
  4. Collect rows that pass ALL predicates → return as CSV

Key SUQL primitives implemented:
  • answer(free_text_field, question_string)  → 'Yes'|'No'|<string>
  • summary(free_text_field)                  → short human-readable summary

Optimizations from the paper (§5):
  • Structured WHERE clauses are evaluated first (cheap), then answer() (expensive).
  • answer() calls are batched and cached to avoid re-querying the same (text, q) pair.
"""

from __future__ import annotations

import re
import textwrap
import json
import hashlib
import sqlite3
import pandas as pd
import os
import time
import httpx

from semantic_dict_context import semantic_guideline
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional
from litellm import completion

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Same LiteLLM/Ollama calling style as test_table_reclamation.py.
# This expects an Ollama-compatible server to be running and reachable at API_BASE.
MODEL = os.environ.get("SUQL_MODEL", "ollama/gemma4:e4b")
API_BASE = os.environ.get("SUQL_API_BASE", "http://localhost:11434")
ANSWER_MAX_TOKENS = int(os.environ.get("SUQL_ANSWER_MAX_TOKENS", "96"))
LOG_RAW_ANSWERS = os.environ.get("SUQL_LOG_RAW_ANSWERS", "0") == "1"
EXPENSIVE_THINK = os.environ.get("EXPENSIVE_THINK", "0") == "1"
EXPENSIVE_NUM_PREDICT = int(
    os.environ.get("EXPENSIVE_NUM_PREDICT", "512" if EXPENSIVE_THINK else "96")
)


def _default_data_path() -> str:
    configured = os.environ.get("LAB_DATA_ROOT")
    if configured:
        return str(Path(configured) / "canonical" / "imdb_joined.csv")
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "data" / "canonical" / "imdb_joined.csv"
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("Cannot find data/canonical/imdb_joined.csv")


DATA_PATH = os.environ.get("SUQL_DATA_PATH") or _default_data_path()

# Table schema exposed to the LLM semantic parser (mirrors the paper's §4 approach)
TABLE_SCHEMA = """
Table: movies
Columns (structured / categorical):
  movie_id  TEXT        -- unique IMDb identifier, e.g. tt0111161
  title     TEXT        -- movie title
  year      INTEGER     -- release year (1895–2011)
  runtime   INTEGER     -- duration in minutes
  director  TEXT        -- director full name
  genres    TEXT        -- comma-separated genre list, e.g. 'Drama,Romance'

Column (unstructured / free text):
  review    TEXT        -- one user review per movie (plain English prose)

SUQL free-text functions available on the 'review' column:
  answer(review, '<yes/no or open question>')
      → 'Yes' | 'No' | short answer string derived from the review text
  summary(review)
      → a brief human-readable summary of the review text
"""

# ---------------------------------------------------------------------------
# Few-shot examples (in-context learning — Section 4 & Appendix A.2 of paper)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = """
-- Example 1
-- Question: What are movies with a runtime under 90 minutes that have funny reviews?
SELECT movie_id, title, year, runtime, director, genres, summary(review) AS review_summary
FROM movies
WHERE runtime < 90
  AND answer(review, 'Is this review funny or does it mention comedy?') = 'Yes'
LIMIT 10;

-- Example 2
-- Question: Top 5 drama movies from the 1990s that reviewers consider masterpieces?
SELECT movie_id, title, year, runtime, director, genres, summary(review) AS review_summary
FROM movies
WHERE genres LIKE '%Drama%'
  AND year >= 1990 AND year <= 1999
  AND answer(review, 'Does the reviewer consider this movie a masterpiece or give it very high praise?') = 'Yes'
ORDER BY year DESC
LIMIT 5;

-- Example 3
-- Question: Find horror movies with terrifying reviews directed by someone named Wes
SELECT movie_id, title, year, runtime, director, genres, summary(review) AS review_summary
FROM movies
WHERE genres LIKE '%Horror%'
  AND director LIKE '%Wes%'
  AND answer(review, 'Does the reviewer find this movie terrifying or scary?') = 'Yes';

-- Example 4
-- Question: What is the longest movie that reviewers found boring?
SELECT movie_id, title, year, runtime, director, genres, summary(review) AS review_summary
FROM movies
WHERE answer(review, 'Does the reviewer find this movie boring or slow-paced?') = 'Yes'
ORDER BY runtime DESC
LIMIT 1;

-- Example 5
-- Question: List 5 comedy movies from 2000 onwards with surprisingly positive reviews
SELECT movie_id, title, year, runtime, director, genres, summary(review) AS review_summary
FROM movies
WHERE genres LIKE '%Comedy%'
  AND year >= 2000
  AND answer(review, 'Is the review surprisingly positive or enthusiastic about the movie?') = 'Yes'
LIMIT 5;
"""

# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

_llm_prompt_count: ContextVar[int | None] = ContextVar("llm_prompt_count", default=None)
_last_query_counts: dict[str, int] = {}


@contextmanager
def _query_metrics_scope(verbose: bool):
    """
    Track query-level metrics for logs.

    Nested scopes reuse the outer counter so ask() includes the parser prompt,
    while direct execute_suql() calls still get their own metrics.
    """
    outermost = _llm_prompt_count.get() is None
    token = None
    start_time = None
    if outermost:
        token = _llm_prompt_count.set(0)
        start_time = time.perf_counter()

    try:
        yield
    finally:
        if outermost:
            elapsed_seconds = time.perf_counter() - start_time
            prompts_sent = _llm_prompt_count.get() or 0
            if verbose:
                print("\nExecution metrics:")
                print(f"  Query execution time: {elapsed_seconds:.2f} seconds")
                print(f"  LLM prompts sent: {prompts_sent}")
            _llm_prompt_count.reset(token)


def _llm_call(system: str, user: str, max_tokens: int = 1024) -> str:
    """
    Single LLM call using LiteLLM, matching the calling style used in
    test_table_reclamation.py.

    The rest of the SUQL engine calls this one function, so replacing this
    backend automatically changes the model used by nl_to_suql(), answer_fn(),
    and summary_fn().
    """
    current_prompt_count = _llm_prompt_count.get()
    if current_prompt_count is not None:
        _llm_prompt_count.set(current_prompt_count + 1)

    response = completion(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=max_tokens,
        api_base=API_BASE,
        timeout=3600,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Step 1: NL → SUQL  (semantic parser, few-shot in-context learning)
# ---------------------------------------------------------------------------

PARSER_SYSTEM = textwrap.dedent(f"""
You are a SUQL (Structured and Unstructured Query Language) semantic parser.
SUQL extends SQL with two free-text functions:
  • answer(free_text_column, 'question')  → 'Yes' | 'No' | short string
  • summary(free_text_column)             → a short prose summary

{TABLE_SCHEMA}

Rules:
1. Use plain SQL WHERE clauses for ALL structured predicates (year, runtime, director, genres, title, movie_id).
2. Use answer() ONLY for predicates that require understanding the review text.
3. Always apply structured filters BEFORE answer() in the WHERE clause (efficiency).
4. The SELECT list must include: movie_id, title, year, runtime, director, genres.
   Add summary(review) AS review_summary whenever you use answer() on the review.
5. Use LIKE '%value%' for partial text matches on genres and director.
6. For "top N" questions without a numeric ranking field, use LIMIT N.
7. Output ONLY the raw SQL query — no markdown fences, no explanation.

Few-shot examples:
{FEW_SHOT_EXAMPLES}
""").strip()


def nl_to_suql(question: str) -> str:
    """
    Translate a natural language question into a SUQL query using
    few-shot in-context learning (Section 4 of the SUQL paper).
    """
    user_prompt = f"-- Question: {question}\n"
    raw = _llm_call(PARSER_SYSTEM, user_prompt, max_tokens=512)
    # Strip any accidental markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\n?```$", "", raw, flags=re.MULTILINE)
    return _extract_first_sql_statement(raw)


# ---------------------------------------------------------------------------
# Step 2: Execute structured SQL on pandas via sqlite3 in-memory
# ---------------------------------------------------------------------------

def _load_dataframe() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["movie_id"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce").fillna(0).astype(int)
    df["review"] = df["review"].fillna("")
    df["title"] = df["title"].fillna("")
    df["director"] = df["director"].fillna("")
    df["genres"] = df["genres"].fillna("")
    return df


def _build_sqlite(df: pd.DataFrame) -> sqlite3.Connection:
    """Load the DataFrame into a temporary in-memory SQLite DB."""
    conn = sqlite3.connect(":memory:")
    df.to_sql("movies", conn, if_exists="replace", index=False)
    return conn


# ---------------------------------------------------------------------------
# Step 3: answer() / summary() — LLM-based free-text operators
# ---------------------------------------------------------------------------

# Simple in-process cache keyed on (review_hash, question) to avoid
# repeat LLM calls for the same (text, question) pair (paper §5.3).
_answer_cache: dict[str, str] = {}

ANSWER_SYSTEM = textwrap.dedent("""
You are a recall-first semantic filter for movie reviews.

Decide whether the review provides evidence that the movie satisfies the question.

Recall is the primary objective, but do not accept every review:
- Answer YES when there is any concrete review-specific evidence: direct wording,
  a synonym/paraphrase, a described example, or a reasonable implication.
- Give borderline evidence the benefit of the doubt when it specifically bears on
  the condition. The evidence need not use the question's exact words.
- This is evidence retrieval, not sentiment scoring: an explicit mention still
  counts when it is negated, qualified, quoted, or used critically.
- Answer NO when there is no review-specific support, the connection is only the
  movie's genre/topic, the text is generic praise or criticism, or the review
  never discusses the condition itself.
- Do not infer evidence merely because the condition is not contradicted.

Return exactly YES or NO. Do not include reasoning, punctuation, or extra text.
""").strip()

SUMMARY_SYSTEM = textwrap.dedent("""
Summarize the following movie review in 1-2 concise sentences.
Capture the reviewer's overall sentiment and the most notable point.
Output only the summary — no preamble.
""").strip()


def _cache_key(text: str, question: str) -> str:
    raw = f"{text}|||{question}"
    return hashlib.md5(raw.encode()).hexdigest()


def answer_fn(review_text: str, question: str) -> str:
    """
    SUQL answer() operator.
    Returns 'Yes', 'No', or a short answer string extracted from review_text.
    """
    key = _cache_key(review_text, question)
    if key in _answer_cache:
        return _answer_cache[key]

    if not review_text or len(review_text.strip()) < 10:
        result = "No"
        raw_result = ""
    else:
        guidance = semantic_guideline(question)
        prompt = (
            (f"{guidance}\n\n" if guidance else "")
            + f"Review:\n{review_text[:6000]}\n\nQuestion: {question}"
        )
        current_prompt_count = _llm_prompt_count.get()
        if current_prompt_count is not None:
            _llm_prompt_count.set(current_prompt_count + 1)
        payload = {
            "model": MODEL.removeprefix("ollama/"),
            "messages": [
                {"role": "system", "content": ANSWER_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": EXPENSIVE_THINK,
            "options": {
                "temperature": 0,
                "num_predict": EXPENSIVE_NUM_PREDICT,
                "num_ctx": 8192,
            },
        }
        result = "No"
        for attempt in range(2):
            try:
                response = httpx.post(
                    f"{API_BASE.rstrip('/')}/api/chat", json=payload, timeout=3600
                )
                response.raise_for_status()
                raw_result = str(response.json().get("message", {}).get("content", ""))
                decision = _parse_decision(raw_result)
                if decision in {"Yes", "No"}:
                    result = decision
                    break
                payload["messages"][-1]["content"] = (
                    prompt + "\n\nYour previous response was invalid. Reply with exactly YES or NO."
                )
            except (httpx.HTTPError, ValueError):
                pass
        if LOG_RAW_ANSWERS:
            print(f"    raw answer() response: {raw_result!r}")

    _answer_cache[key] = result
    return result


def _parse_decision(raw: str) -> str | None:
    """Parse a model decision without relying on provider-side schemas."""
    text = str(raw).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            text = str(parsed.get("decision", parsed.get("answer", text)))
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    match = re.search(r"\b(uncertain|yes|no)\b", text, re.IGNORECASE)
    if not match or match.group(1).lower() == "uncertain":
        return None
    return match.group(1).capitalize()


def summary_fn(review_text: str) -> str:
    """SUQL summary() operator — returns a short prose summary of the review."""
    if not review_text or len(review_text.strip()) < 10:
        return "(no review)"
    key = _cache_key(review_text, "__summary__")
    if key in _answer_cache:
        return _answer_cache[key]
    result = " ".join(_llm_call(SUMMARY_SYSTEM, review_text[:2000], max_tokens=80).split())
    _answer_cache[key] = result
    return result


def _write_metrics_sidecar(
    engine_seconds: float,
    result_rows: int,
    llm_full_calls: int,
    llm_early_accept: int = 0,
    llm_early_reject: int = 0,
) -> None:
    metrics_path = os.environ.get("SUQL_METRICS_PATH")
    if not metrics_path:
        return
    payload = {
        "engine_seconds": float(engine_seconds),
        "llm_full_calls": int(llm_full_calls),
        "llm_prompts_issued": int(llm_full_calls),
        "llm_early_accept": int(llm_early_accept),
        "llm_early_reject": int(llm_early_reject),
        "result_rows": int(result_rows),
        "structured_candidates": int(_last_query_counts.get("structured_candidates", 0)),
        "semantic_rows": int(_last_query_counts.get("semantic_rows", result_rows)),
        "join_rows": int(_last_query_counts.get("join_rows", result_rows)),
        "nonempty_fallback_rows": int(_last_query_counts.get("nonempty_fallback_rows", 0)),
    }
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Step 4: SUQL query execution — parse, split, execute
# ---------------------------------------------------------------------------

# Regex to find answer() and summary() calls in the query
_ANSWER_RE = re.compile(
    r"answer\s*\(\s*(\w+)\s*,\s*(['\"])(.*?)\2\s*\)",
    re.IGNORECASE | re.DOTALL,
)
_SUMMARY_SELECT_RE = re.compile(
    r"summary\s*\(\s*(\w+)\s*\)",
    re.IGNORECASE,
)


def _extract_first_sql_statement(text: str) -> str:
    """
    Keep only the first SELECT statement from an LLM response.
    Local models sometimes append notes or extra few-shot examples.
    """
    match = re.search(r"\bSELECT\b", text, flags=re.IGNORECASE)
    if not match:
        return text.strip()

    sql = text[match.start():]
    in_quote: str | None = None
    for i, char in enumerate(sql):
        if char in ("'", '"'):
            if in_quote == char:
                in_quote = None
            elif in_quote is None:
                in_quote = char
        elif char == ";" and in_quote is None:
            return sql[: i + 1].strip()

    lines = []
    for line in sql.splitlines():
        if line.startswith(("###", "Note:", "-- Question:", "User:", "System:")):
            break
        lines.append(line)
    return "\n".join(lines).strip()


def _extract_answer_predicates(sql: str) -> list[tuple[str, str, str]]:
    """
    Return list of (full_match, column, question) for every answer() call in the SQL.
    """
    return [(m.group(0), m.group(1), m.group(3)) for m in _ANSWER_RE.finditer(sql)]


def _strip_answer_predicates(sql: str) -> str:
    """
    Remove answer() calls from WHERE clause so the query can run on SQLite.
    Handles:  AND answer(...)='Yes'   /   answer(...) = 'Yes' AND   /   WHERE answer(...)='Yes'
    """
    # Remove  AND answer(...)='<val>'  or  answer(...)='<val>' AND
    cleaned = re.sub(
        r"\bAND\s+answer\s*\([^)]+\)\s*=\s*['\"][^'\"]*['\"]",
        "",
        sql,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"answer\s*\([^)]+\)\s*=\s*['\"][^'\"]*['\"\s]*AND\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Remove standalone  WHERE answer(...)='<val>'
    cleaned = re.sub(
        r"\bWHERE\s+answer\s*\([^)]+\)\s*=\s*['\"][^'\"]*['\"]",
        "WHERE 1=1",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned


def _strip_summary_from_select(sql: str) -> str:
    """
    Replace  summary(col) AS alias  with  col  in the SELECT list
    so SQLite can run the structural query.  We'll compute summary later.
    """
    return re.sub(
        r"summary\s*\(\s*(\w+)\s*\)\s+AS\s+\w+",
        r"\1",
        sql,
        flags=re.IGNORECASE,
    )


def _strip_summary_order_by(sql: str) -> str:
    """
    Remove ORDER BY summary(col) clauses before running SQLite.
    summary() is evaluated after structural SQL, so SQLite cannot sort by it.
    """
    return re.sub(
        r"\bORDER\s+BY\s+summary\s*\(\s*\w+\s*\)\s*(?:ASC|DESC)?\s*(?=\bLIMIT\b|;|$)",
        "",
        sql,
        flags=re.IGNORECASE,
    )


def _strip_function_order_by(sql: str) -> str:
    """
    Remove ORDER BY clauses that call unsupported functions such as decade(year).
    SQLite can still sort by real columns, but SUQL functions are post-processed.
    """
    return re.sub(
        r"\bORDER\s+BY\s+\w+\s*\([^)]*\)\s*(?:ASC|DESC)?\s*(?=\bLIMIT\b|;|$)",
        "",
        sql,
        flags=re.IGNORECASE,
    )


def _get_limit(sql: str) -> Optional[int]:
    match = re.search(r"\bLIMIT\s+(\d+)\s*;?\s*$", sql, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _strip_limit(sql: str) -> str:
    return re.sub(r"\bLIMIT\s+\d+\s*;?\s*$", "", sql, flags=re.IGNORECASE).rstrip() + ";"


def _select_list_match(sql: str) -> Optional[re.Match[str]]:
    return re.search(r"^\s*SELECT\s+(.*?)\s+FROM\s+", sql, flags=re.IGNORECASE | re.DOTALL)


def _get_selected_output_names(sql: str) -> set[str]:
    """Return simple output column names/aliases from the SELECT list."""
    match = _select_list_match(sql)
    if not match:
        return set()

    names: set[str] = set()
    for item in match.group(1).split(","):
        item = item.strip()
        alias_match = re.search(r"\s+AS\s+(\w+)\s*$", item, flags=re.IGNORECASE)
        if alias_match:
            names.add(alias_match.group(1).lower())
            continue

        simple_col = item.split(".")[-1].strip()
        if re.fullmatch(r"\w+", simple_col):
            names.add(simple_col.lower())

    return names


def _ensure_columns_selected(sql: str, columns: set[str]) -> str:
    """
    Add internal columns needed by answer()/summary() to the SELECT list.
    They are removed from the final result if the user did not request them.
    """
    if not columns:
        return sql

    match = _select_list_match(sql)
    if not match:
        return sql

    select_list = match.group(1)
    if select_list.strip() == "*":
        return sql

    selected = _get_selected_output_names(sql)
    missing = [col for col in sorted(columns) if col.lower() not in selected]
    if not missing:
        return sql

    insert_at = match.end(1)
    return f"{sql[:insert_at]}, {', '.join(missing)}{sql[insert_at:]}"


def _get_answer_expected(sql: str) -> dict[tuple[str, str], str]:
    """
    Build a mapping  {(column, question): expected_value}  from answer() predicates.
    e.g.  answer(review, 'Is it amazing?') = 'Yes'  →  {('review', 'Is it amazing?'): 'Yes'}
    """
    result: dict[tuple[str, str], str] = {}
    pattern = re.compile(
        r"answer\s*\(\s*(\w+)\s*,\s*(['\"])(.*?)\2\s*\)\s*=\s*(['\"])(.*?)\4",
        re.IGNORECASE | re.DOTALL,
    )
    for m in pattern.finditer(sql):
        col, question, expected = m.group(1), m.group(3), m.group(5)
        result[(col, question)] = expected
    return result


def _get_summary_columns(sql: str) -> dict[str, str]:
    """
    Return {alias: column} for each  summary(col) AS alias  in SELECT.
    """
    result: dict[str, str] = {}
    pattern = re.compile(
        r"summary\s*\(\s*(\w+)\s*\)\s+AS\s+(\w+)",
        re.IGNORECASE,
    )
    for m in pattern.finditer(sql):
        result[m.group(2)] = m.group(1)
    return result


def _execute_suql_impl(
    suql_query: str,
    df: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Execute a SUQL query and return a DataFrame of results.

    Pipeline (mirrors the SUQL paper §5):
      1. Split into structural SQL (for SQLite) + free-text predicates (answer/summary).
      2. Run structural SQL on SQLite → candidate rows.
      3. For each candidate, evaluate answer() predicates via LLM (cached).
      4. Keep only rows that pass all answer() predicates.
      5. Compute summary() values for requested columns.
      6. Return final DataFrame.
    """
    if df is None:
        df = _load_dataframe()

    _last_query_counts.clear()
    suql_query = _extract_first_sql_statement(suql_query)

    if verbose:
        print(f"\n{'='*60}")
        print("SUQL Query:")
        print(suql_query)
        print(f"{'='*60}")

    # --- Parse answer() and summary() predicates ---
    answer_predicates = _get_answer_expected(suql_query)
    summary_cols = _get_summary_columns(suql_query)
    final_limit = _get_limit(suql_query) if answer_predicates else None
    requested_output_names = _get_selected_output_names(suql_query)
    internal_cols = {col for col, _question in answer_predicates}
    internal_cols.update(summary_cols.values())
    internal_only_cols = {
        col for col in internal_cols
        if col.lower() not in requested_output_names
    }

    # --- Strip free-text ops, run structural SQL on SQLite ---
    structural_sql = _strip_answer_predicates(suql_query)
    structural_sql = _strip_summary_from_select(structural_sql)
    structural_sql = _strip_summary_order_by(structural_sql)
    structural_sql = _strip_function_order_by(structural_sql)
    if final_limit is not None:
        structural_sql = _strip_limit(structural_sql)
    structural_sql = _ensure_columns_selected(structural_sql, internal_cols)

    conn = _build_sqlite(df)
    try:
        candidate_df = pd.read_sql_query(structural_sql, conn)
    except Exception as exc:
        conn.close()
        raise RuntimeError(
            f"SQLite execution failed.\nQuery:\n{structural_sql}\nError: {exc}"
        ) from exc
    conn.close()

    if verbose:
        print(f"Structural filter → {len(candidate_df)} candidate rows")
    _last_query_counts["structured_candidates"] = len(candidate_df)

    if candidate_df.empty:
        return candidate_df
    # --- Apply answer() predicates (LLM, cached) ---
    if answer_predicates:
        keep_mask = pd.Series([True] * len(candidate_df), index=candidate_df.index)

        for (col, question), expected in answer_predicates.items():
            if col not in candidate_df.columns:
                if verbose:
                    print(f"  [warn] Column '{col}' not found — skipping answer() predicate")
                continue
            if verbose:
                print(f"  answer({col}, '{question[:60]}...') = '{expected}' — evaluating {len(candidate_df[keep_mask])} rows…")

            active_indexes = list(candidate_df[keep_mask].index)
            total_active = len(active_indexes)
            for position, idx in enumerate(active_indexes, start=1):
                text = str(candidate_df.at[idx, col])
                result = answer_fn(text, question)
                if result.strip().lower() != expected.strip().lower():
                    keep_mask.at[idx] = False
                if verbose and (position % 25 == 0 or position == total_active):
                    kept_so_far = int(keep_mask.loc[active_indexes[:position]].sum())
                    print(f"    checked {position}/{total_active} rows, kept {kept_so_far}")

        candidate_df = candidate_df[keep_mask].copy()
        if verbose:
            print(f"  After answer() filter → {len(candidate_df)} rows")
        _last_query_counts["semantic_rows"] = len(candidate_df)
    else:
        _last_query_counts["semantic_rows"] = len(candidate_df)

    if final_limit is not None:
        candidate_df = candidate_df.head(final_limit)
    _last_query_counts["join_rows"] = len(candidate_df)

    # --- Compute summary() columns ---
    for alias, col in summary_cols.items():
        if col in candidate_df.columns:
            if verbose:
                print(f"  Computing summary({col}) for {len(candidate_df)} rows…")
            candidate_df[alias] = candidate_df[col].apply(
                lambda t: summary_fn(str(t))
            )
        else:
            candidate_df[alias] = ""

    # Drop columns added only so answer()/summary() could run.
    if internal_only_cols:
        candidate_df = candidate_df.drop(columns=list(internal_only_cols), errors="ignore")

    return candidate_df.reset_index(drop=True)


def execute_suql(
    suql_query: str,
    df: Optional[pd.DataFrame] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Execute a SUQL query and log query-level metrics when verbose."""
    with _query_metrics_scope(verbose):
        return _execute_suql_impl(suql_query, df=df, verbose=verbose)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _ask_impl(
    question: str,
    output_csv: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline:
      NL question → SUQL query → execute → return DataFrame (and optionally save CSV).

    Parameters
    ----------
    question   : Natural language question about the movie database.
    output_csv : If provided, save results to this CSV file path.
    verbose    : Print progress information.

    Returns
    -------
    pd.DataFrame with the query results.
    """
    if verbose:
        print(f"\n{'#'*60}")
        print(f"Question: {question}")
        print(f"{'#'*60}")

    # Step 1: NL → SUQL
    suql_query = nl_to_suql(question)
    if verbose:
        print(f"\n[Semantic Parser] Generated SUQL:\n{suql_query}\n")

    # Step 2–5: Execute SUQL
    df = _load_dataframe()
    results = execute_suql(suql_query, df=df, verbose=verbose)

    if verbose:
        print(f"\nResults ({len(results)} rows):")
        print(results.to_string(index=False))

    # Save CSV
    if output_csv:
        results.to_csv(output_csv, index=False)
        if verbose:
            print(f"\nSaved → {output_csv}")

    return results


def ask(
    question: str,
    output_csv: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run a natural-language query and log query-level metrics when verbose."""
    with _query_metrics_scope(verbose):
        return _ask_impl(question, output_csv=output_csv, verbose=verbose)


def _ask_with_suql_impl(
    suql_query: str,
    output_csv: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Execute a manually-written SUQL query directly (skip NL parsing).

    Parameters
    ----------
    suql_query : A SUQL query string.
    output_csv : If provided, save results to this CSV file path.
    verbose    : Print progress information.

    Returns
    -------
    pd.DataFrame with the query results.
    """
    start_time = time.perf_counter()
    df = _load_dataframe()
    results = execute_suql(suql_query, df=df, verbose=verbose)
    engine_seconds = time.perf_counter() - start_time

    if output_csv:
        results.to_csv(output_csv, index=False)
        if verbose:
            print(f"\nSaved → {output_csv}")

    _write_metrics_sidecar(
        engine_seconds=engine_seconds,
        result_rows=len(results),
        llm_full_calls=_llm_prompt_count.get() or 0,
    )
    return results


def ask_with_suql(
    suql_query: str,
    output_csv: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run a raw SUQL query and log query-level metrics when verbose."""
    with _query_metrics_scope(verbose):
        return _ask_with_suql_impl(suql_query, output_csv=output_csv, verbose=verbose)
