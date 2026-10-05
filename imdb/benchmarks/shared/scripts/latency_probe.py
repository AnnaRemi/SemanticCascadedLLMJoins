"""Per-model latency, log-prob availability and cheap-score signal probe.

Runs against a live Ollama server using the SUQL v1 cheap scorer and
expensive-answer code (so it measures what the benchmark actually does).

For every model it reports
  * seconds per cheap scoring call and per expensive answer call (sequential),
  * whether /v1/completions and the native /api/generate return real log-probs,
  * how many distinct cheap scores came back (3 or fewer means the scorer has
    silently fallen back to a +-2 yes/no label),
  * ROC-AUC of the cheap score against the benchmark ground truth.

Usage: latency_probe.py --api-base URL --suql-dir DIR --data FILE --out FILE
                        --models a,b,c [--questions q_01,q_02,q_05]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request


def post(url: str, payload: dict, timeout: float = 600.0) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def auc(scores: list[float], labels: list[int]) -> float | None:
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "mean": statistics.mean(values),
        "p50": ordered[len(ordered) // 2],
        "p90": ordered[int(0.9 * (len(ordered) - 1))],
        "n": len(values),
    }


def raw_logprob_check(api_base: str, model: str, review: str, question: str) -> dict:
    prompt = f"Question: {question}\nReview: {review[:600]}\n\nAnswer 1 for Yes, 0 for No:"
    report: dict = {}
    try:
        payload = post(
            f"{api_base}/v1/completions",
            {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0,
             "logprobs": 10, "think": False},
        )
        choice = (payload.get("choices") or [{}])[0]
        report["v1_completions_logprobs_present"] = bool(choice.get("logprobs"))
        report["v1_completions_sample"] = json.dumps(choice)[:400]
    except Exception as exc:  # noqa: BLE001
        report["v1_completions_error"] = f"{type(exc).__name__}: {exc}"[:300]
    try:
        payload = post(
            f"{api_base}/api/generate",
            {"model": model, "prompt": prompt, "stream": False, "think": False,
             "logprobs": True, "top_logprobs": 10, "options": {"temperature": 0, "num_predict": 1}},
        )
        report["native_generate_logprobs_present"] = bool(payload.get("logprobs"))
        report["native_generate_sample"] = json.dumps(payload.get("logprobs"))[:400]
    except Exception as exc:  # noqa: BLE001
        report["native_generate_error"] = f"{type(exc).__name__}: {exc}"[:300]
    return report


def probe_model(args: argparse.Namespace, bundle: dict, model: str) -> dict:
    from cascade_filter import CascadeAnswerFilter

    flt = CascadeAnswerFilter(
        cheap_model=f"ollama/{model}",
        expensive_model=f"ollama/{model}",
        api_base=args.api_base,
        manual_confidence_threshold=None,
    )
    questions = [q for q in args.questions.split(",") if q in bundle]
    first = bundle[questions[0]]
    result: dict = {"model": model}

    started = time.perf_counter()
    flt.cheap_scorer.score(first["rows"][0]["review"], first["semantic_question"])
    result["load_and_first_call_seconds"] = time.perf_counter() - started
    result["raw_logprobs"] = raw_logprob_check(
        args.api_base, model, first["rows"][0]["review"], first["semantic_question"]
    )
    result["scorer_used_v1_completions"] = flt.cheap_scorer._openai_completions_available

    scores: list[float] = []
    labels: list[int] = []
    cheap_times: list[float] = []
    for name in questions:
        for row in bundle[name]["rows"]:
            tic = time.perf_counter()
            try:
                score = float(flt.cheap_scorer.score(row["review"], bundle[name]["semantic_question"]))
            except Exception as exc:  # noqa: BLE001
                result.setdefault("score_errors", []).append(f"{type(exc).__name__}: {exc}"[:200])
                continue
            cheap_times.append(time.perf_counter() - tic)
            scores.append(score)
            labels.append(row["gt"])
    if cheap_times:
        result["cheap_call_seconds"] = summarize(cheap_times)
        result["distinct_scores"] = len(set(round(s, 4) for s in scores))
        result["score_min"], result["score_max"] = min(scores), max(scores)
        result["auc_vs_ground_truth"] = auc(scores, labels)
        negative_floor = min(scores)
        positives = [s for s, y in zip(scores, labels) if y == 1]
        if positives:
            result["share_gt_positives_at_lowest_score"] = sum(s == negative_floor for s in positives) / len(positives)

    expensive_times: list[float] = []
    agree = []
    for name in questions[:2]:
        for row in bundle[name]["rows"][:15]:
            tic = time.perf_counter()
            answer = flt.expensive_answer(row["review"], bundle[name]["semantic_question"])
            expensive_times.append(time.perf_counter() - tic)
            agree.append((answer == "Yes") == bool(row["gt"]))
    if expensive_times:
        result["expensive_call_seconds"] = summarize(expensive_times)
        result["expensive_agreement_with_ground_truth"] = sum(agree) / len(agree)

    try:  # free the GPU for the next model
        post(f"{args.api_base}/api/generate", {"model": model, "keep_alive": 0}, timeout=120)
    except Exception:  # noqa: BLE001
        pass
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--suql-dir", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--questions", default="q_01,q_02,q_05")
    args = parser.parse_args()

    sys.path.insert(0, args.suql_dir)
    bundle = json.load(open(args.data))
    results: list[dict] = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"=== probing {model}", flush=True)
        try:
            results.append(probe_model(args, bundle, model))
        except Exception as exc:  # noqa: BLE001
            results.append({"model": model, "error": f"{type(exc).__name__}: {exc}"[:500]})
        print(json.dumps(results[-1], indent=1), flush=True)
        json.dump(results, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
