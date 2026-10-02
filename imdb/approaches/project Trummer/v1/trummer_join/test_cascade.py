"""Light tests for Trummer v1's Beta-posterior cascade calibration (paper
sec:cascade-math). No real Ollama server is reachable in this sandbox (no
cross-process loopback networking), so the integration test runs a tiny mock
HTTP server as a thread in this same process and points CascadeJoin at it.
See threshold_fit.py for the stdlib-only Beta(1,1)-credible-lower-bound fit.
"""
from __future__ import annotations

import json
import os
import re
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

# These tests don't ship a semantic_dict.json; run the semantic-guidance layer
# in its documented soft-failure mode instead of requiring one.
os.environ.setdefault("SEMANTIC_DICT_REQUIRED", "0")

from .cascade import CascadeConfig, CascadeJoin
from .threshold_fit import beta_lower_bound, fit_cascade_threshold


class TestFitCascadeThreshold(unittest.TestCase):
    def test_separable_classes_yield_a_confident_band_on_both_sides(self) -> None:
        # Same n=30-per-side sizing verified against profiler.py's numpy/scipy
        # twin: beta_lower_bound(29, 0, 0.9) is the first n to clear 0.9.
        scores = [5.0] * 30 + [-5.0] * 30
        labels = [1] * 30 + [0] * 30
        result = fit_cascade_threshold(
            "q", scores, labels, accept_precision_target=0.9,
            reject_precision_target=0.9, credible_level=0.9,
        )
        self.assertLess(result.cheap_accept_threshold, float("inf"))
        self.assertGreater(result.cheap_reject_threshold, float("-inf"))
        self.assertGreaterEqual(result.accept_precision_lower, 0.9)
        self.assertGreaterEqual(result.reject_precision_lower, 0.9)
        self.assertEqual(result.expensive_count, 0)

    def test_matches_the_numpy_scipy_twin_in_profiler_py(self) -> None:
        # threshold_fit.py is an intentional stdlib-only duplicate of
        # project SUQL/v1/profiler.py; the two must agree numerically.
        self.assertAlmostEqual(
            beta_lower_bound(18, 2, 0.9),
            0.7659531683480946,  # scipy.stats.beta.ppf(0.1, 19, 3), computed once and pinned
            places=9,
        )


def _extract_ids(text: str, prefix: str) -> list[int]:
    return [int(value) for value in re.findall(rf"{prefix}_(\d+):", text)]


class _MockOllama(BaseHTTPRequestHandler):
    """/api/chat only: cheap batches are addressed as CANDIDATE_<id>, expensive
    batches as PAIR_<id> (see cascade.py's _cheap_batch_prompt /
    _expensive_batch_prompt). Decisions are looked up by candidate_id."""

    cheap_labels: dict[int, str] = {}
    oracle_accepts: dict[int, bool] = {}

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        pass

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        payload = json.loads(raw or b"{}")
        content = str(payload.get("messages", [{}])[-1].get("content", ""))

        cheap_ids = _extract_ids(content, "CANDIDATE")
        pair_ids = _extract_ids(content, "PAIR")
        lines = [
            f"CANDIDATE_{cid}: {self.cheap_labels.get(cid, 'NO')}" for cid in cheap_ids
        ] + [
            f"PAIR_{cid}: {'YES' if self.oracle_accepts.get(cid, False) else 'NO'}"
            for cid in pair_ids
        ]
        body = json.dumps({"message": {"content": "\n".join(lines)}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _movies_and_reviews(n: int) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    movies = [{"movie_id": f"m{i}", "text": f"Movie {i}"} for i in range(n)]
    reviews = [{"tconst": f"m{i}", "text": f"Review text {i}"} for i in range(n)]
    return movies, reviews


class TestCascadeJoinEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = HTTPServer(("127.0.0.1", 0), _MockOllama)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.thread.join(timeout=5)

    def test_beta_calibration_resolves_confident_pairs_without_escalating(self) -> None:
        # candidate_ids are assigned in review order by exact_id_candidates:
        # 1..30 are the "positive" reviews, 31..60 the "negative" ones.
        movies, reviews = _movies_and_reviews(60)
        _MockOllama.cheap_labels = {i: "YES" for i in range(1, 31)} | {
            i: "NO" for i in range(31, 61)
        }
        _MockOllama.oracle_accepts = {i: True for i in range(1, 31)} | {
            i: False for i in range(31, 61)
        }

        config = CascadeConfig(
            api_base=f"http://127.0.0.1:{self.port}",
            cascade_target=0.9,
            credible_level=0.9,
            calibration_budget=60,
            request_timeout=5.0,
        )
        rows, decisions, metrics = CascadeJoin(config).run(movies, reviews, "predicate")

        accepted_movie_ids = {row["movie_id"] for row in rows}
        self.assertEqual(accepted_movie_ids, {f"m{i - 1}" for i in range(1, 31)})
        self.assertTrue(metrics.calibration_activated)
        self.assertIsNotNone(metrics.learned_accept_threshold)
        self.assertIsNotNone(metrics.learned_reject_threshold)
        self.assertGreaterEqual(metrics.calibration_accept_precision_lower, 0.9)
        self.assertGreaterEqual(metrics.calibration_reject_precision_lower, 0.9)
        # Nothing should have needed the expensive fallback past calibration.
        self.assertEqual(metrics.expensive_candidates, 0)

    def test_uncertain_pair_is_escalated_to_the_expensive_stage(self) -> None:
        # A clearly-separable calibration set, plus one held-out pair the
        # cheap stage calls UNCERTAIN -- it must be escalated and resolved by
        # the expensive oracle rather than silently defaulted.
        movies, reviews = _movies_and_reviews(61)
        _MockOllama.cheap_labels = {i: "YES" for i in range(1, 31)} | {
            i: "NO" for i in range(31, 61)
        }
        _MockOllama.cheap_labels[61] = "UNCERTAIN"
        _MockOllama.oracle_accepts = {i: True for i in range(1, 31)} | {
            i: False for i in range(31, 61)
        }
        _MockOllama.oracle_accepts[61] = True  # resolved YES by the expensive stage

        config = CascadeConfig(
            api_base=f"http://127.0.0.1:{self.port}",
            cascade_target=0.9,
            credible_level=0.9,
            calibration_budget=60,
            request_timeout=5.0,
        )
        rows, decisions, metrics = CascadeJoin(config).run(movies, reviews, "predicate")

        accepted_movie_ids = {row["movie_id"] for row in rows}
        self.assertIn("m60", accepted_movie_ids)  # candidate 61 -> movie m60
        fence_decision = next(d for d in decisions if d.candidate_id == 61)
        self.assertEqual(fence_decision.route, "expensive")


if __name__ == "__main__":
    unittest.main()
