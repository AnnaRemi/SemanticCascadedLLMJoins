"""Tests for the stratified Beta-posterior band fit (band_fit.py)."""
from __future__ import annotations

import math
import random
import unittest

from .band_fit import (
    allocate_labels,
    fit_band,
    stratified_sample,
    stratum_edges,
    stratum_of,
)


class TestStrata(unittest.TestCase):
    def test_equal_count_strata_on_distinct_scores(self) -> None:
        scores = [float(i) for i in range(40)]
        edges = stratum_edges(scores, 4)
        self.assertEqual(edges, [9.0, 19.0, 29.0])
        counts = [0] * 4
        for s in scores:
            counts[stratum_of(s, edges)] += 1
        self.assertEqual(counts, [10, 10, 10, 10])

    def test_ties_never_straddle_two_strata(self) -> None:
        # yes/unsure/no style scores: the cut points must land on tie boundaries.
        scores = [-2.0] * 27 + [0.0] * 5 + [2.0] * 8
        edges = stratum_edges(scores, 4)
        self.assertEqual(edges, [-2.0, 0.0])
        self.assertEqual({stratum_of(s, edges) for s in scores}, {0, 1, 2})
        for value in (-2.0, 0.0, 2.0):
            self.assertEqual(len({stratum_of(s, edges) for s in scores if s == value}), 1)

    def test_single_value_has_one_stratum(self) -> None:
        self.assertEqual(stratum_edges([1.0] * 10, 4), [])


class TestSampling(unittest.TestCase):
    def test_allocation_spends_the_budget_and_respects_stratum_size(self) -> None:
        self.assertEqual(allocate_labels([10, 10, 10, 10], 20), [5, 5, 5, 5])
        alloc = allocate_labels([27, 1, 12], 20)
        self.assertEqual(sum(alloc), 20)
        self.assertLessEqual(alloc[1], 1)
        self.assertEqual(allocate_labels([3, 3], 20), [3, 3])

    def test_every_stratum_gets_labels(self) -> None:
        scores = [float(i) for i in range(40)]
        edges = stratum_edges(scores, 4)
        picks = stratified_sample(scores, 20, 4)
        self.assertEqual(len(picks), 20)
        self.assertEqual(sorted({stratum_of(scores[i], edges) for i in picks}), [0, 1, 2, 3])

    def test_seeded_sampling_varies_by_seed_and_is_reproducible(self) -> None:
        scores = [float(i) for i in range(200)]
        first = stratified_sample(scores, 20, 4, random.Random("a"))
        again = stratified_sample(scores, 20, 4, random.Random("a"))
        other = stratified_sample(scores, 20, 4, random.Random("b"))
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)

    def test_budget_covering_the_pool_labels_everything(self) -> None:
        self.assertEqual(stratified_sample([0.0, 1.0, 2.0], 5), [0, 1, 2])


class TestFitBand(unittest.TestCase):
    def _labels(self, truth: list[int], picks: list[int]) -> dict[int, int]:
        return {i: truth[i] for i in picks}

    def test_separable_pool_certifies_both_sides(self) -> None:
        scores = [-5.0] * 100 + [5.0] * 100
        truth = [0] * 100 + [1] * 100
        picks = stratified_sample(scores, 40, 4)
        fit = fit_band(scores, self._labels(truth, picks), quality_target=0.8, credible_level=0.9)
        self.assertTrue(fit.activated)
        self.assertGreater(fit.reject_threshold, -math.inf)
        self.assertLess(fit.accept_threshold, math.inf)
        self.assertGreaterEqual(fit.reject_recall_lower, 0.8)
        self.assertGreaterEqual(fit.accept_precision_lower, 0.8)
        self.assertLess(fit.reject_threshold, fit.accept_threshold)

    def test_rare_positives_that_the_score_does_not_rank_are_not_rejected(self) -> None:
        # The IMDb collapse: 2 positives in 40 rows, hidden mid-range. Rejecting a
        # whole prefix of strata could lose them, so nothing may be certified.
        scores = [float((i * 7) % 13 - 6) + i * 1e-3 for i in range(40)]
        truth = [1 if i in (5, 17) else 0 for i in range(40)]
        picks = stratified_sample(scores, 20, 4)
        fit = fit_band(scores, self._labels(truth, picks), quality_target=0.8, credible_level=0.9)
        self.assertFalse(fit.activated)
        self.assertEqual(fit.reject_threshold, -math.inf)

    def test_nothing_is_decided_when_every_row_is_labelled(self) -> None:
        scores = [-5.0] * 10 + [5.0] * 10
        truth = [0] * 10 + [1] * 10
        fit = fit_band(scores, dict(enumerate(truth)), quality_target=0.8, credible_level=0.9)
        self.assertFalse(fit.activated)
        self.assertEqual(fit.rejected_unlabelled + fit.accepted_unlabelled, 0)

    def test_empty_inputs_are_safe(self) -> None:
        self.assertFalse(fit_band([], {}, quality_target=0.8, credible_level=0.9).activated)
        self.assertFalse(fit_band([1.0, 2.0], {}, quality_target=0.8, credible_level=0.9).activated)

    def test_fit_is_deterministic(self) -> None:
        rng = random.Random(3)
        scores = [rng.gauss(0, 1) for _ in range(300)]
        truth = [1 if s + rng.gauss(0, 1) > 1 else 0 for s in scores]
        picks = stratified_sample(scores, 40, 4)
        a = fit_band(scores, self._labels(truth, picks), quality_target=0.8, credible_level=0.9)
        b = fit_band(scores, self._labels(truth, picks), quality_target=0.8, credible_level=0.9)
        self.assertEqual(a, b)

    def test_certified_reject_regions_keep_their_recall_in_simulation(self) -> None:
        # Credible-bound sanity check: over many simulated pools, whenever a reject
        # region is certified for Q* = 0.8 the true recall must reach 0.8 almost always
        # (alpha = 0.9 allows up to 10% misses; the estimator is conservative).
        rng = random.Random(11)
        certified = kept = 0
        for _ in range(60):
            scores, truth = [], []
            for _ in range(200):
                y = 1 if rng.random() < 0.3 else 0
                truth.append(y)
                scores.append(rng.gauss(2.0 * y, 1.0))
            picks = stratified_sample(scores, 40, 4, rng)
            fit = fit_band(scores, self._labels(truth, picks), quality_target=0.8, credible_level=0.9)
            if fit.reject_threshold == -math.inf:
                continue
            certified += 1
            labelled = set(picks)
            positives = sum(truth)
            lost = sum(1 for i, (s, y) in enumerate(zip(scores, truth))
                       if y and s <= fit.reject_threshold and i not in labelled)
            kept += (1 - lost / positives) >= 0.8
        self.assertGreater(certified, 20)
        self.assertGreaterEqual(kept / certified, 0.9)


if __name__ == "__main__":
    unittest.main()
