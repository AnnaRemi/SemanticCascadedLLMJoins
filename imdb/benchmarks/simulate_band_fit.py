#!/usr/bin/env python3
"""Simulations behind Sec. 4.4 of the paper: validity and power of the stratified
Beta-posterior band fit (band_fit.py).

For synthetic pools with Gaussian cheap scores (positives shifted by ``sep``, so
AUC = Phi(sep / sqrt 2)) it reports, per configuration,
  * validity: among trials where a reject region was certified at Q*, the share
    whose TRUE recall (positives not lost, labelled rows being answered exactly)
    reached Q*; the target is at least alpha;
  * power: the share of the pool the cheap stage decides on its own.

Usage: simulate_band_fit.py [--trials 300] [--quality 0.8] [--alpha 0.9]
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "approaches" / "project SUQL" / "v1"))
from band_fit import fit_band, stratified_sample  # noqa: E402


def make_pool(rng: random.Random, n: int, prevalence: float, sep: float):
    scores, truth = [], []
    for _ in range(n):
        y = 1 if rng.random() < prevalence else 0
        scores.append(rng.gauss(sep * y, 1.0))
        truth.append(y)
    return scores, truth


def trial(rng, n, prevalence, sep, budget, quality, alpha, strata=4):
    scores, truth = make_pool(rng, n, prevalence, sep)
    picks = stratified_sample(scores, budget, strata, rng)
    fit = fit_band(scores, {i: truth[i] for i in picks}, quality_target=quality,
                   credible_level=alpha, n_strata=strata)
    labelled = set(picks)
    out = {"decided": (fit.rejected_unlabelled + fit.accepted_unlabelled) / n, "certified": False}
    if fit.reject_threshold > -math.inf:
        positives = sum(truth)
        lost = sum(1 for i, (s, y) in enumerate(zip(scores, truth))
                   if y and s <= fit.reject_threshold and i not in labelled)
        out["certified"] = True
        out["recall_ok"] = (1 - lost / positives if positives else 1.0) >= quality
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--quality", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.9)
    args = parser.parse_args()
    rng = random.Random(1)
    print("pool  prev  AUC   |C|  K  reject certified   true recall >= Q* (target >= %.2f)   share of pool decided" % args.alpha)
    configs = [
        (40, .3, 1.0, 20, 4), (40, .3, 2.0, 20, 4), (40, .1, 2.0, 20, 4),
        (200, .3, 2.0, 20, 4), (200, .3, 2.0, 40, 4),
        (1000, .3, 2.5, 20, 4), (1000, .3, 2.5, 40, 4), (1000, .3, 2.5, 100, 8),
    ]
    for n, prev, sep, budget, strata in configs:
        runs = [trial(rng, n, prev, sep, budget, args.quality, args.alpha, strata) for _ in range(args.trials)]
        cert = [r for r in runs if r["certified"]]
        auc = 0.5 * (1 + math.erf(sep / 2.0))
        valid = (sum(r["recall_ok"] for r in cert) / len(cert)) if cert else float("nan")
        decided = sum(r["decided"] for r in runs) / len(runs)
        print(f"{n:<5} {prev:<5} {auc:.2f}  {budget:<4} {strata:<2} {len(cert):>3}/{args.trials:<3}          {valid:5.3f}                                  {decided:.2f}")


if __name__ == "__main__":
    main()
