"""Stratified Beta-posterior fit of the cascade's accept/reject band.

Paper Sec. 4.4 chooses the reject threshold tau_- so that a credible lower bound
on the *recall* of the cheap stage stays at or above the quality target Q*, and
the accept threshold tau_+ so that a credible lower bound on *precision* does.
This module implements that rule for a calibration set that is drawn
*stratified by cheap score* instead of uniformly at random.

Why stratified: with |C| = 20 and a positive rate of 5-30 percent, a uniform
sample holds 1-6 positives, far too few to certify a recall of 0.8 (that needs
about ten positives with no misses). Sampling a fixed number of rows from each
score stratum puts labels where the thresholds are decided. The price is that
the sample no longer mirrors the pool, so a raw Beta(1+TP, 1+FN) on the sample
would be biased; instead each stratum gets its own Beta(1+y, 1+n-y) posterior
for its positive rate, and the posterior for recall/precision of a candidate
band is obtained by Monte Carlo over those independent posteriors, including
the finite-population draw for the rows that were not labelled.

Rows that *were* labelled are answered with their oracle label (they are not
routed again), so only unlabelled rows can be lost by a reject or wrongly
accepted by an accept.

Stdlib only, so it can be shipped unchanged into the Trummer package.
"""
from __future__ import annotations

import bisect
import math
import random
from dataclasses import asdict, dataclass
from typing import Sequence

DEFAULT_STRATA = 4
DEFAULT_DRAWS = 2000
_MC_SEED = 20260805


@dataclass(frozen=True)
class BandFit:
    accept_threshold: float  # math.inf: accept nothing without verification
    reject_threshold: float  # -math.inf: reject nothing without verification
    accept_precision_lower: float  # 0.0 when no accept region was certified
    reject_recall_lower: float  # 0.0 when no reject region was certified
    strata_sizes: tuple[int, ...]
    strata_labelled: tuple[int, ...]
    strata_positives: tuple[int, ...]
    rejected_unlabelled: int
    accepted_unlabelled: int

    @property
    def activated(self) -> bool:
        return self.accept_threshold < math.inf or self.reject_threshold > -math.inf

    def to_json_dict(self) -> dict:
        return asdict(self)


def stratum_edges(scores: Sequence[float], n_strata: int = DEFAULT_STRATA) -> list[float]:
    """Inclusive upper score bound of every stratum but the last.

    Strata are equal-count slices of the sorted scores, with each cut moved to
    the next tie boundary so that equal scores never straddle two strata (the
    router compares scores against thresholds, so ties must route alike). With
    coarse scores (e.g. yes/unsure/no) this yields fewer than ``n_strata``.
    """
    ordered = sorted(scores)
    n = len(ordered)
    edges: list[float] = []
    start = 0
    for h in range(1, max(1, n_strata)):
        cut = max(round(h * n / n_strata), start + 1)
        while cut < n and ordered[cut] == ordered[cut - 1]:
            cut += 1
        if cut >= n:
            break
        edges.append(ordered[cut - 1])
        start = cut
    return edges


def stratum_of(score: float, edges: Sequence[float]) -> int:
    return bisect.bisect_left(edges, score)


def allocate_labels(sizes: Sequence[int], budget: int) -> list[int]:
    """Spread ``budget`` labels as evenly as possible over strata, capped by
    stratum size; leftovers go to the highest-scoring strata first."""
    alloc = [0] * len(sizes)
    remaining = max(0, budget)
    while remaining > 0:
        active = [h for h in range(len(sizes)) if alloc[h] < sizes[h]]
        if not active:
            break
        share = max(1, remaining // len(active))
        for h in sorted(active, reverse=True):
            give = min(share, sizes[h] - alloc[h], remaining)
            alloc[h] += give
            remaining -= give
            if remaining == 0:
                break
    return alloc


def stratified_sample(
    scores: Sequence[float],
    budget: int,
    n_strata: int = DEFAULT_STRATA,
    rng: random.Random | None = None,
) -> list[int]:
    """Indices into ``scores`` of the rows to label.

    ``rng`` None gives the deterministic evenly spaced pick inside each
    stratum; a Random instance draws uniformly inside each stratum (the
    per-repetition CALIBRATION_SEED path).
    """
    if budget <= 0 or not scores:
        return []
    if len(scores) <= budget:
        return list(range(len(scores)))
    edges = stratum_edges(scores, n_strata)
    members: list[list[int]] = [[] for _ in range(len(edges) + 1)]
    for index, score in enumerate(scores):
        members[stratum_of(score, edges)].append(index)
    for group in members:
        group.sort(key=lambda i: (scores[i], i))
    chosen: list[int] = []
    for group, take in zip(members, allocate_labels([len(g) for g in members], budget)):
        if take <= 0:
            continue
        if rng is not None:
            chosen.extend(rng.sample(group, take))
        elif take == 1:
            chosen.append(group[len(group) // 2])
        else:
            chosen.extend(group[round(i * (len(group) - 1) / (take - 1))] for i in range(take))
    return sorted(chosen)


def _binomial(rng: random.Random, n: int, p: float) -> int:
    if n <= 0 or p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    if n <= 64:
        return sum(rng.random() < p for _ in range(n))
    mean = n * p
    return min(n, max(0, round(rng.gauss(mean, math.sqrt(mean * (1.0 - p))))))


def _lower(values: list[float], credible_level: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((1.0 - credible_level) * len(ordered)))]


def fit_band(
    scores: Sequence[float],
    labels: dict[int, int],
    *,
    quality_target: float,
    credible_level: float,
    n_strata: int = DEFAULT_STRATA,
    draws: int = DEFAULT_DRAWS,
) -> BandFit:
    """Fit the accept/reject band from oracle ``labels`` (index -> 0/1) on a
    pool with cheap ``scores``.

    Reject: the widest score prefix of strata whose credible lower bound on
    recall of the final answer set (1 - lost positives / all positives) is at
    least ``quality_target``. Accept: the widest score suffix whose credible
    lower bound on the precision of the unverified accepted rows is at least
    ``quality_target``. Anything not certified is escalated to the expensive
    model, so a weak calibration set can only cost money, never quality.
    """
    n = len(scores)
    edges = stratum_edges(scores, n_strata)
    k = len(edges) + 1
    sizes = [0] * k
    labelled = [0] * k
    positives = [0] * k
    for index, score in enumerate(scores):
        sizes[stratum_of(score, edges)] += 1
    for index, label in labels.items():
        h = stratum_of(scores[index], edges)
        labelled[h] += 1
        positives[h] += int(label == 1)
    unlabelled = [sizes[h] - labelled[h] for h in range(k)]
    known_positives = sum(positives)

    no_band = BandFit(
        math.inf, -math.inf, 0.0, 0.0, tuple(sizes), tuple(labelled), tuple(positives), 0, 0
    )
    if n == 0 or k < 2 or not labels:
        return no_band

    rng = random.Random(_MC_SEED)
    recall_draws: list[list[float]] = [[] for _ in range(k)]  # index b = strata rejected
    precision_draws: list[list[float]] = [[] for _ in range(k + 1)]  # index t = first accepted
    for _ in range(draws):
        extra = [
            _binomial(rng, unlabelled[h], rng.betavariate(1 + positives[h], 1 + labelled[h] - positives[h]))
            for h in range(k)
        ]
        total = known_positives + sum(extra)
        lost = 0
        for b in range(1, k):
            lost += extra[b - 1]
            recall_draws[b].append(1.0 - lost / total if total > 0 else 1.0)
        accepted_positive = 0
        accepted_rows = 0
        for t in range(k - 1, 0, -1):
            accepted_positive += extra[t]
            accepted_rows += unlabelled[t]
            if accepted_rows > 0:
                precision_draws[t].append(accepted_positive / accepted_rows)

    reject_strata = 0
    reject_lower = 0.0
    for b in range(k - 1, 0, -1):
        if sum(unlabelled[:b]) == 0:
            continue
        lower = _lower(recall_draws[b], credible_level)
        if lower >= quality_target:
            reject_strata, reject_lower = b, lower
            break

    accept_from = k
    accept_lower = 0.0
    for t in range(max(reject_strata, 1), k):
        if not precision_draws[t]:
            continue
        lower = _lower(precision_draws[t], credible_level)
        if lower >= quality_target:
            accept_from, accept_lower = t, lower
            break

    reject_threshold = edges[reject_strata - 1] if reject_strata else -math.inf
    accept_threshold = math.inf
    if accept_from < k:
        accept_threshold = min(s for s in scores if stratum_of(s, edges) >= accept_from)
    return BandFit(
        accept_threshold=accept_threshold,
        reject_threshold=reject_threshold,
        accept_precision_lower=accept_lower,
        reject_recall_lower=reject_lower,
        strata_sizes=tuple(sizes),
        strata_labelled=tuple(labelled),
        strata_positives=tuple(positives),
        rejected_unlabelled=sum(unlabelled[:reject_strata]),
        accepted_unlabelled=sum(unlabelled[accept_from:]) if accept_from < k else 0,
    )
