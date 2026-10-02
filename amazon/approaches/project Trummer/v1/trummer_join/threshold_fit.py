from __future__ import annotations

import dataclasses
import math

# Intentional stdlib-only duplicate of the core fitting logic in
# "project SUQL/v1/profiler.py" (which uses numpy/scipy). Trummer v1's
# requirements.txt is Python-standard-library-only by design, so this port
# reimplements the same Beta(1,1)-credible-lower-bound threshold search
# without numpy/scipy. Keep the two in sync if the fitting algorithm changes.


@dataclasses.dataclass(frozen=True)
class CascadeThreshold:
    question: str
    cheap_accept_threshold: float
    cheap_reject_threshold: float
    accept_precision_lower: float
    reject_precision_lower: float
    accept_precision_target: float
    reject_precision_target: float
    credible_level: float
    labelled_examples: int
    early_accept_count: int
    early_reject_count: int
    expensive_count: int

    def to_json_dict(self) -> dict:
        return dataclasses.asdict(self)


def _log_binom_pmf(n: int, k: int, log_x: float, log_1mx: float) -> float:
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * log_x
        + (n - k) * log_1mx
    )


def _binom_tail_prob(n: int, a: int, x: float) -> float:
    """P(Y >= a) for Y ~ Binomial(n, x). Used to evaluate the Beta(a, b) CDF
    at x via the classical identity CDF_Beta(x; a, b) = P(Binomial(a+b-1, x) >= a)
    for positive integer a, b."""
    if a <= 0:
        return 1.0
    if a > n:
        return 0.0
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_x = math.log(x)
    log_1mx = math.log1p(-x)
    term = math.exp(_log_binom_pmf(n, a, log_x, log_1mx))
    total = term
    ratio_base = math.exp(log_x - log_1mx)
    for k in range(a, n):
        term *= (n - k) / (k + 1) * ratio_base
        total += term
        if term < 1e-15 * max(total, 1e-300):
            break
    return min(total, 1.0)


def beta_lower_bound(successes: int, failures: int, credible_level: float) -> float:
    """Bayesian Beta(1,1)-prior credible lower bound on precision.

    Equivalent to scipy.stats.beta.ppf(1 - credible_level, 1 + successes, 1 + failures),
    solved by bisection against the Beta-CDF/Binomial-tail identity so this
    stays stdlib-only (see project SUQL/v1/profiler.py for the scipy original).
    """
    a = 1 + successes
    n = successes + failures + 1
    target = 1.0 - credible_level

    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _binom_tail_prob(n, a, mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


def _candidate_thresholds(scores: list[float]) -> list[float]:
    finite = sorted({s for s in scores if math.isfinite(s)})
    if not finite:
        raise ValueError("No finite scores available.")
    lo, hi = finite[0], finite[-1]
    if hi > lo:
        grid = [lo + (hi - lo) * i / 199 for i in range(200)]
    else:
        grid = [lo]
    return sorted(set(finite) | set(grid))


def _fit_accept_threshold(
    scores: list[float],
    labels: list[int],
    precision_target: float,
    credible_level: float,
) -> tuple[float, float, int]:
    best: tuple[float, float, int] | None = None
    for threshold in _candidate_thresholds(scores):
        tp = 0
        fp = 0
        for score, label in zip(scores, labels):
            if score >= threshold:
                if label == 1:
                    tp += 1
                else:
                    fp += 1
        selected_count = tp + fp
        if selected_count == 0:
            continue
        lower = beta_lower_bound(tp, fp, credible_level)
        if lower < precision_target:
            continue
        candidate = (float(threshold), lower, selected_count)
        if best is None or candidate[2] > best[2] or (candidate[2] == best[2] and candidate[0] < best[0]):
            best = candidate
    if best is None:
        return math.inf, 1.0, 0
    return best


def _fit_reject_threshold(
    scores: list[float],
    labels: list[int],
    precision_target: float,
    credible_level: float,
) -> tuple[float, float, int]:
    best: tuple[float, float, int] | None = None
    for threshold in _candidate_thresholds(scores):
        tn = 0
        fn = 0
        for score, label in zip(scores, labels):
            if score <= threshold:
                if label == 0:
                    tn += 1
                else:
                    fn += 1
        selected_count = tn + fn
        if selected_count == 0:
            continue
        lower = beta_lower_bound(tn, fn, credible_level)
        if lower < precision_target:
            continue
        candidate = (float(threshold), lower, selected_count)
        if best is None or candidate[2] > best[2] or (candidate[2] == best[2] and candidate[0] > best[0]):
            best = candidate
    if best is None:
        return -math.inf, 1.0, 0
    return best


@dataclasses.dataclass(frozen=True)
class RejectThreshold:
    threshold: float  # math.inf if nothing could be confidently rejected
    precision_lower: float
    rejected_count: int
    kept_count: int


def fit_reject_threshold(
    scores: list[float],
    labels: list[int],
    precision_target: float = 0.9,
    credible_level: float = 0.9,
) -> RejectThreshold:
    """Find the most permissive reject-only threshold whose Beta-credible lower
    bound on reject precision still clears ``precision_target``.

    Deliberately conservative: with too few or too ambiguous examples it
    returns ``threshold=inf`` (reject nothing) rather than over-pruning.
    """
    threshold, precision_lower, rejected_count = _fit_reject_threshold(
        scores, labels, precision_target, credible_level
    )
    return RejectThreshold(
        threshold=threshold,
        precision_lower=precision_lower,
        rejected_count=rejected_count,
        kept_count=len(scores) - rejected_count,
    )


def _median(sorted_values: list[float]) -> float:
    n = len(sorted_values)
    mid = n // 2
    if n % 2 == 1:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def fit_cascade_threshold(
    question: str,
    scores: list[float],
    labels: list[int],
    accept_precision_target: float = 0.9,
    reject_precision_target: float = 0.9,
    credible_level: float = 0.9,
) -> CascadeThreshold:
    accept_threshold, accept_lower, early_accept_count = _fit_accept_threshold(
        scores,
        labels,
        precision_target=accept_precision_target,
        credible_level=credible_level,
    )
    reject_threshold, reject_lower, early_reject_count = _fit_reject_threshold(
        scores,
        labels,
        precision_target=reject_precision_target,
        credible_level=credible_level,
    )
    if reject_threshold >= accept_threshold:
        # Keep a real unsure band if independently fitted thresholds cross.
        midpoint = _median(sorted(scores))
        reject_threshold = min(reject_threshold, midpoint)
        accept_threshold = max(accept_threshold, midpoint)
        early_accept_count = sum(1 for s in scores if s >= accept_threshold)
        early_reject_count = sum(1 for s in scores if s <= reject_threshold)

    expensive_count = len(scores) - early_accept_count - early_reject_count
    return CascadeThreshold(
        question=question,
        cheap_accept_threshold=accept_threshold,
        cheap_reject_threshold=reject_threshold,
        accept_precision_lower=accept_lower,
        reject_precision_lower=reject_lower,
        accept_precision_target=accept_precision_target,
        reject_precision_target=reject_precision_target,
        credible_level=credible_level,
        labelled_examples=len(scores),
        early_accept_count=early_accept_count,
        early_reject_count=early_reject_count,
        expensive_count=max(expensive_count, 0),
    )
