"""Light tests for the Beta-posterior cascade calibration (paper sec:cascade-math).

No real Ollama server is reachable in this sandbox (no cross-process loopback
networking), so the integration test runs a tiny mock HTTP server as a thread in
this same process and points CascadeAnswerFilter at it. See profiler.py for the
Beta(1,1)-credible-lower-bound fit itself.
"""
from __future__ import annotations

import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

# These tests don't ship a semantic_dict.json; run the semantic-guidance layer
# in its documented soft-failure mode instead of requiring one.
os.environ.setdefault("SEMANTIC_DICT_REQUIRED", "0")

from cascade_filter import CascadeAnswerFilter
from profiler import beta_lower_bound, fit_cascade_threshold


class TestBetaLowerBound(unittest.TestCase):
    def test_matches_scipy_directly(self) -> None:
        from scipy.stats import beta

        lower = beta_lower_bound(successes=18, failures=2, credible_level=0.9)
        expected = float(beta.ppf(0.1, 19, 3))
        self.assertAlmostEqual(lower, expected, places=9)

    def test_more_evidence_pulls_bound_closer_to_point_estimate(self) -> None:
        # Same 90% point estimate (9/10 vs 90/100), but more evidence should
        # give a tighter (higher) credible lower bound.
        small = beta_lower_bound(9, 1, credible_level=0.9)
        large = beta_lower_bound(90, 10, credible_level=0.9)
        self.assertLess(small, large)
        self.assertLess(small, 0.9)
        self.assertLess(large, 0.9)

    def test_no_evidence_is_uninformative(self) -> None:
        # Beta(1,1) prior with zero observations: the bound is exactly 1-credible_level.
        self.assertAlmostEqual(beta_lower_bound(0, 0, 0.9), 0.1, places=9)


class TestFitCascadeThreshold(unittest.TestCase):
    def test_separable_classes_yield_a_confident_band_on_both_sides(self) -> None:
        # 30 clear positives (score 5), 30 clear negatives (score -5). At
        # target=credible_level=0.9 this needs >=~29 perfect agreements per
        # side (beta_lower_bound(n, 0, 0.9) only clears 0.9 at n=29): verified
        # empirically that n=20 (0.896) falls just short and n=30 (0.928) clears.
        scores = np.array([5.0] * 30 + [-5.0] * 30)
        labels = np.array([1] * 30 + [0] * 30)
        result = fit_cascade_threshold(
            "q", scores, labels, accept_precision_target=0.9,
            reject_precision_target=0.9, credible_level=0.9,
        )
        self.assertLess(result.cheap_accept_threshold, float("inf"))
        self.assertGreater(result.cheap_reject_threshold, float("-inf"))
        self.assertGreaterEqual(result.accept_precision_lower, 0.9)
        self.assertGreaterEqual(result.reject_precision_lower, 0.9)
        self.assertLessEqual(result.cheap_reject_threshold, result.cheap_accept_threshold)
        # Every row should be confidently resolved; nothing left to escalate.
        self.assertEqual(result.expensive_count, 0)

    def test_small_sample_cannot_clear_a_strict_target_even_with_perfect_agreement(
        self,
    ) -> None:
        # 10-per-side perfect agreement sounds great by raw accuracy, but the
        # Beta(1,1) posterior correctly refuses to certify 90%-credible
        # precision from only 10 examples (lower bound ~0.81 < 0.9): this is
        # the paper's "self-correcting for evidence" property, not a bug.
        scores = np.array([5.0] * 10 + [-5.0] * 10)
        labels = np.array([1] * 10 + [0] * 10)
        result = fit_cascade_threshold(
            "q", scores, labels, accept_precision_target=0.9,
            reject_precision_target=0.9, credible_level=0.9,
        )
        self.assertEqual(result.cheap_accept_threshold, float("inf"))
        self.assertEqual(result.cheap_reject_threshold, float("-inf"))
        self.assertEqual(result.expensive_count, 20)

    def test_ambiguous_tiny_sample_stays_open_rather_than_overclaiming(self) -> None:
        # A single mixed-signal pair can't support a 90%-credible claim either way.
        scores = np.array([0.1, -0.1])
        labels = np.array([1, 0])
        result = fit_cascade_threshold(
            "q", scores, labels, accept_precision_target=0.9,
            reject_precision_target=0.9, credible_level=0.9,
        )
        self.assertEqual(result.cheap_accept_threshold, float("inf"))
        self.assertEqual(result.cheap_reject_threshold, float("-inf"))
        self.assertEqual(result.expensive_count, 2)

    def test_crossing_thresholds_are_corrected_to_keep_a_real_band(self) -> None:
        # scores=[0,1,2], labels=[1,0,1] with a lenient 0.5 target: the accept
        # side's best cutoff (0.0, keeping all 3 rows) and the reject side's
        # best cutoff (just under 2.0, keeping the first two) overlap, so
        # fit_cascade_threshold must pull both back to the shared median (1.0)
        # rather than return an inverted band. Verified empirically.
        scores = np.array([0.0, 1.0, 2.0])
        labels = np.array([1, 0, 1])
        result = fit_cascade_threshold(
            "q", scores, labels, accept_precision_target=0.5,
            reject_precision_target=0.5, credible_level=0.5,
        )
        self.assertGreaterEqual(result.cheap_accept_threshold, result.cheap_reject_threshold)
        self.assertEqual(result.cheap_accept_threshold, 1.0)
        self.assertEqual(result.cheap_reject_threshold, 1.0)


def _extract_marker(text: str, markers: list[str]) -> str | None:
    for marker in markers:
        if marker in text:
            return marker
    return None


class _MockOllama(BaseHTTPRequestHandler):
    """Routes /v1/completions -> 404 (forces the native fallback), /api/generate
    -> a cheap log-odds score, /api/chat -> an oracle YES/NO, both looked up by
    a review-text marker embedded in the request body."""

    cheap_scores: dict[str, float] = {}
    oracle_labels: dict[str, str] = {}

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - stdlib signature
        pass

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw or b"{}")

    def _reply(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler name
        if self.path == "/v1/completions":
            self._reply(404, {"error": "not found"})
            return
        payload = self._body()
        if self.path == "/api/generate":
            prompt = str(payload.get("prompt", ""))
            marker = _extract_marker(prompt, list(self.cheap_scores))
            score = self.cheap_scores.get(marker, 0.0) if marker else 0.0
            # values["1"] - values["0"] == score, exactly, per scorer._mapping_score.
            self._reply(200, {"logprobs": {"1": score, "0": 0.0}})
            return
        if self.path == "/api/chat":
            content = str(payload.get("messages", [{}])[-1].get("content", ""))
            marker = _extract_marker(content, list(self.oracle_labels))
            label = self.oracle_labels.get(marker, "No") if marker else "No"
            self._reply(200, {"message": {"content": label}})
            return
        self._reply(404, {"error": "unknown path"})


class TestCascadeAnswerFilterEndToEnd(unittest.TestCase):
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

    def test_beta_calibration_routes_confident_rows_without_escalating(self) -> None:
        # 30 clearly-positive and 30 clearly-negative reviews; calibration_budget
        # covers all of them. 30-per-side is needed (not 20) because the Beta(1,1)
        # posterior only certifies 90%-credible precision from ~29+ perfect
        # agreements (verified in TestFitCascadeThreshold) -- so the Beta band
        # should resolve every row from the cheap score alone here.
        # Markers must be >= 10 chars: answer_batch auto-rejects shorter text
        # as a non-review before the cheap scorer ever runs.
        reviews = [f"POSITIVE_REVIEW_{i:03d}" for i in range(30)] + [
            f"NEGATIVE_REVIEW_{i:03d}" for i in range(30)
        ]
        _MockOllama.cheap_scores = {m: 5.0 for m in reviews[:30]} | {
            m: -5.0 for m in reviews[30:]
        }
        _MockOllama.oracle_labels = {m: "Yes" for m in reviews[:30]} | {
            m: "No" for m in reviews[30:]
        }

        cascade = CascadeAnswerFilter(
            api_base=f"http://127.0.0.1:{self.port}",
            cheap_model="mock-cheap",
            expensive_model="mock-expensive",
            cascade_target=0.9,
            credible_level=0.9,
            calibration_budget=60,
            timeout=5.0,
            request_retries=1,
        )
        answers = cascade.answer_batch(reviews, "does it have evidence?")

        self.assertEqual(answers[:30], ["Yes"] * 30)
        self.assertEqual(answers[30:], ["No"] * 30)
        usage = cascade.stats.model_usage_by_question["does it have evidence?"]
        self.assertEqual(usage["calibration_mode"], "oracle")
        self.assertTrue(usage["calibration_activated"])
        self.assertGreaterEqual(usage["calibration_accept_precision_lower"], 0.9)
        self.assertGreaterEqual(usage["calibration_reject_precision_lower"], 0.9)

    def test_ambiguous_row_is_escalated_to_the_expensive_oracle(self) -> None:
        # Calibration set is clearly separable (so a real band is learned), but
        # one held-out row sits exactly on the fence and must be escalated.
        calibration_reviews = [f"CALIBRATION_POSITIVE_{i:03d}" for i in range(30)] + [
            f"CALIBRATION_NEGATIVE_{i:03d}" for i in range(30)
        ]
        fence_review = "THE_FENCE_SITTING_ROW"
        reviews = calibration_reviews + [fence_review]

        _MockOllama.cheap_scores = {m: 5.0 for m in calibration_reviews[:30]}
        _MockOllama.cheap_scores.update({m: -5.0 for m in calibration_reviews[30:]})
        _MockOllama.cheap_scores[fence_review] = 0.0  # inside any sane band
        _MockOllama.oracle_labels = {m: "Yes" for m in calibration_reviews[:30]}
        _MockOllama.oracle_labels.update({m: "No" for m in calibration_reviews[30:]})
        _MockOllama.oracle_labels[fence_review] = "Yes"

        cascade = CascadeAnswerFilter(
            api_base=f"http://127.0.0.1:{self.port}",
            cheap_model="mock-cheap",
            expensive_model="mock-expensive",
            cascade_target=0.9,
            credible_level=0.9,
            calibration_budget=60,
            timeout=5.0,
            request_retries=1,
        )
        answers = cascade.answer_batch(reviews, "is the fence row evidence present?")
        self.assertEqual(answers[-1], "Yes")  # resolved via the expensive oracle
        usage = cascade.stats.model_usage_by_question[
            "is the fence row evidence present?"
        ]
        self.assertGreater(usage["expensive_full_calls"], len(calibration_reviews))


if __name__ == "__main__":
    unittest.main()
