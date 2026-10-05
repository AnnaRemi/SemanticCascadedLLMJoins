from __future__ import annotations

import csv
import json
import os
import resource
from pathlib import Path


# A handful of Amazon Fashion rows have scraped JS/HTML dumped into a field
# (observed up to ~422KB), which blows past Python's default 128KB csv limit.
csv.field_size_limit(10_000_000)

LAB_ROOT = Path(__file__).resolve().parents[3]
# The four method implementations live under <dataset>/approaches/.
APPROACH_ROOT = LAB_ROOT / "approaches"
SUITE_ROOT = Path(os.environ["BENCHMARK_SUITE_ROOT"]).resolve()
QUESTION_ROOT = SUITE_ROOT / "per_question"


def _data_root() -> Path:
    configured = os.environ.get("LAB_DATA_ROOT")
    if configured:
        return Path(configured).resolve()
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "data"
        if (candidate / "subdatasets").exists():
            return candidate
    return LAB_ROOT / "data"


DATA_ROOT = _data_root()
SUBDATASET_ROOT = DATA_ROOT / "subdatasets" / SUITE_ROOT.name


def question_dir(name: str) -> Path:
    path = QUESTION_ROOT / name
    if not (path / "benchmark.json").exists():
        raise FileNotFoundError(f"Unknown question directory: {path}")
    return path


def question_data_dir(name: str) -> Path:
    # Metadata stays with the benchmark suite; all CSV inputs are centralized.
    question_dir(name)
    path = SUBDATASET_ROOT / name
    if not (path / "amazon_reviews.csv").exists():
        raise FileNotFoundError(f"Missing centralized question data: {path}")
    return path


def benchmark(name: str) -> dict:
    return json.loads((question_dir(name) / "benchmark.json").read_text())


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def load_products(name: str) -> list[dict[str, str]]:
    rows = []
    with (question_data_dir(name) / "amazon_structured_joined.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            row["text"] = (
                f"product_id={row.get('product_id', '')}; title={row.get('title', '')}; "
                f"year={row.get('year', '')}; director={row.get('director', '')}; "
                f"runtime={row.get('runtime', '')}; genres={row.get('genres', '')}"
            )
            rows.append(row)
    return rows


def load_reviews(name: str) -> list[dict[str, str]]:
    rows = []
    with (question_data_dir(name) / "amazon_reviews.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            review = " ".join((row.get("review") or "").replace("<br />", " ").split())
            row["review"] = review
            row["text"] = f"tconst={row.get('tconst', '')}; review={review[:1800]}"
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = ["product_id"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
