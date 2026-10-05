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
    # Simulates a context overflow: batches above this many pairs get no decisions.
    max_pairs: int | None = None

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        pass

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        payload = json.loads(raw or b"{}")
        content = str(payload.get("messages", [{}])[-1].get("content", ""))

        cheap_ids = _extract_ids(content, "CANDIDATE")
        pair_ids = _extract_ids(content, "PAIR")
        if self.max_pairs is not None and len(pair_ids) > self.max_pairs:
            pair_ids = []
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
    movies = [{"product_id": f"m{i}", "text": f"Product {i}"} for i in range(n)]
    reviews = [{"product_id": f"m{i}", "text": f"Review text {i}"} for i in range(n)]
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

    def _run(self, movies, reviews, budget: int):
        config = CascadeConfig(
            api_base=f"http://127.0.0.1:{self.port}",
            cascade_target=0.8,
            credible_level=0.9,
            calibration_budget=budget,
            request_timeout=5.0,
        )
        return CascadeJoin(config).run(movies, reviews, "predicate")

    def test_confident_pairs_are_resolved_without_escalating_and_labels_are_reused(self) -> None:
        # candidate_ids are assigned in review order by exact_id_candidates:
        # 1..60 are the "positive" reviews, 61..120 the "negative" ones. Only 40
        # pairs are labelled by the oracle; the other 80 must be resolved by the
        # cheap stage, and the 40 labelled ones must keep their oracle answer.
        movies, reviews = _movies_and_reviews(120)
        _MockOllama.cheap_labels = {i: "YES" for i in range(1, 61)} | {i: "NO" for i in range(61, 121)}
        _MockOllama.oracle_accepts = {i: True for i in range(1, 61)} | {i: False for i in range(61, 121)}

        rows, decisions, metrics = self._run(movies, reviews, budget=40)

        self.assertEqual({row["product_id"] for row in rows}, {f"m{i - 1}" for i in range(1, 61)})
        self.assertTrue(metrics.calibration_activated)
        self.assertIsNotNone(metrics.learned_accept_threshold)
        self.assertIsNotNone(metrics.learned_reject_threshold)
        self.assertGreaterEqual(metrics.calibration_accept_precision_lower, 0.8)
        self.assertGreaterEqual(metrics.calibration_reject_recall_lower, 0.8)
        self.assertEqual(metrics.calibration_reused_labels, 40)
        self.assertEqual(metrics.expensive_candidates, 0)
        # Every expensive call was a calibration call: nothing was escalated afterwards.
        self.assertEqual(metrics.expensive_calls, metrics.calibration_expensive_calls)
        routes = [d.route for d in decisions]
        self.assertEqual(routes.count("calibration_accept") + routes.count("calibration_reject"), 40)

    def test_uncertain_pairs_are_escalated_and_answered_by_the_oracle(self) -> None:
        # Clearly separable blocks plus 40 UNCERTAIN pairs whose oracle label is a
        # coin flip: the cheap stage cannot certify them, so the unlabelled ones
        # must be escalated and resolved by the expensive stage.
        movies, reviews = _movies_and_reviews(160)
        _MockOllama.cheap_labels = {i: "YES" for i in range(1, 61)} | {i: "NO" for i in range(61, 121)}
        _MockOllama.cheap_labels |= {i: "UNCERTAIN" for i in range(121, 161)}
        _MockOllama.oracle_accepts = {i: True for i in range(1, 61)} | {i: False for i in range(61, 121)}
        _MockOllama.oracle_accepts |= {i: i % 2 == 0 for i in range(121, 161)}

        rows, decisions, metrics = self._run(movies, reviews, budget=40)

        expected = {f"m{i - 1}" for i in range(1, 161) if _MockOllama.oracle_accepts[i]}
        self.assertEqual({row["product_id"] for row in rows}, expected)
        self.assertGreater(metrics.expensive_candidates, 0)
        self.assertTrue(any(d.route == "expensive" for d in decisions))
        # No pair is paid for twice.
        self.assertEqual(
            metrics.calibration_reused_labels + metrics.expensive_candidates
            + metrics.cheap_early_accepts + metrics.cheap_early_rejects,
            160,
        )

    def test_a_failed_expensive_batch_is_split_and_retried_not_dropped(self) -> None:
        # Regression: when an expensive batch came back unusable (context overflow
        # makes the model omit candidate ids), the whole batch was silently dropped, so
        # real positives vanished. The batch must be bisected until the pieces fit.
        movies, reviews = _movies_and_reviews(40)
        _MockOllama.cheap_labels = {i: "UNCERTAIN" for i in range(1, 41)}
        _MockOllama.oracle_accepts = {i: i <= 10 for i in range(1, 41)}
        _MockOllama.max_pairs = 8
        try:
            rows, _decisions, metrics = self._run(movies, reviews, budget=20)
        finally:
            _MockOllama.max_pairs = None

        self.assertEqual({row["product_id"] for row in rows}, {f"m{i - 1}" for i in range(1, 11)})
        self.assertGreater(metrics.expensive_failures, 0)
        self.assertEqual(metrics.expensive_undecided, 0)
        # Retries are real calls and must be counted.
        self.assertGreater(metrics.expensive_calls, 3)

    def test_rare_positives_never_collapse_the_cascade_into_rejecting_everything(self) -> None:
        # Regression for the IMDb collapse: with only 2 oracle positives in 40 pairs,
        # a reject test on precision passes trivially and the cheap stage drops every
        # pair. The cheap labels here do not single out the positives, so a recall
        # guarantee cannot be certified and the oracle's answers must be returned.
        movies, reviews = _movies_and_reviews(40)
        _MockOllama.cheap_labels = {i: ("NO", "UNCERTAIN", "YES")[i % 3] for i in range(1, 41)}
        _MockOllama.oracle_accepts = {i: i in (6, 18) for i in range(1, 41)}

        rows, _decisions, metrics = self._run(movies, reviews, budget=20)

        self.assertEqual({row["product_id"] for row in rows}, {"m5", "m17"})
        self.assertFalse(metrics.calibration_activated)
        self.assertEqual(metrics.cheap_early_rejects, 0)


if __name__ == "__main__":
    unittest.main()
