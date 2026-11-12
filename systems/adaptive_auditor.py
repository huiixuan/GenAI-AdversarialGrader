"""Adaptive Auditor — System 4.

Pipeline for one question
-------------------------
1. SOLVE   — ask the model once (standard SOLVE_PROMPT).
2. SAMPLE  — draw (SAMPLE_K - 1) more answers at solver temperature to build
             a mini-ensemble.  Combined with the first answer this gives
             SAMPLE_K predictions.
3. RISK    — compute a risk score ∈ [0, 1]:
               risk = 1 - (frequency of the modal answer / SAMPLE_K)
             A unanimous ensemble → risk = 0.
             A completely split ensemble → risk close to 1.
4. AUDIT   — if risk >= RISK_THRESHOLD, send the question + the model's
             first answer to an auditor LLM that checks the working and
             either confirms or overrides the answer.
5. VERDICT — return the final answer (audited or majority-vote), plus
             metadata (risk score, whether the auditor fired, cost).

Tuning knobs (all in this file, change them on dev.json freely)
---------------------------------------------------------------
SAMPLE_K        : total predictions per question  (default 5)
RISK_THRESHOLD  : audit when risk >= this value   (default 0.4)
AUDIT_PROMPT    : the auditor's instruction
"""

import concurrent.futures
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (AUDITOR_MODEL, AUDITOR_TEMP, RESULTS_DIR, SOLVER_MODEL, SOLVER_TEMP,
                    WORKERS, shared_run_id)
from grading import extract_model_answer, is_correct
from llm import Tracker, call_llm, check_ready
from prompts import SOLVE_PROMPT
from systems.common import solve_once

# ── tuning knobs ─────────────────────────────────────────────────────────────
SAMPLE_K_INITIAL  = 3    # samples to draw first before checking risk
SAMPLE_K_EXTRA    = 2    # additional samples if risk is borderline
RISK_LOW          = 0.3  # below this → trust majority vote, no audit
RISK_HIGH         = 0.5  # above this → audit immediately, skip extra samples
# borderline zone: RISK_LOW <= risk <= RISK_HIGH → draw SAMPLE_K_EXTRA more, then re-check
RISK_THRESHOLD    = 0.4  # audit threshold applied after the final sample set

AUDIT_PROMPT_BLIND_SOLVE = (
    "You are an independent math auditor. Solve the following problem yourself "
    "without any bias. Show your key steps concisely.\n\n"
    "Problem:\n{question}\n\n"
    "Finish with a single line in exactly this form, with nothing after it:\n"
    "ANSWER: <your answer as a number or simple fraction, no units>"
)

AUDIT_PROMPT_COMPARE = (
    "You solved the problem and got: {auditor_answer}\n"
    "The solver's answer is: {solver_answer}\n\n"
    "Only override the solver if you are highly confident it is wrong. "
    "If there is any doubt, return the solver's answer.\n\n"
    "Finish with a single line in exactly this form, with nothing after it:\n"
    "ANSWER: <final answer as a number or simple fraction, no units>"
)

# ── result dataclass ──────────────────────────────────────────────────────────
@dataclass
class AuditResult:
    question_id: str
    predicted: str | None    # final answer after audit (or majority vote)
    gold: str
    correct: bool
    tag: str                 # from dataset: seeded_error / consistent_correct / always_wrong

    # auditor metadata
    risk_score: float           # 0.0 = unanimous, 1.0 = maximally split
    auditor_triggered: bool     # did we pay for a second call?
    majority_answer: str | None # majority-vote answer BEFORE audit
    majority_correct: bool      # was the pre-audit answer already correct?
    auditor_answer: str | None  # what the auditor said (None if not triggered)

    # cost metadata
    calls: int
    in_tokens: int
    out_tokens: int
    latency_s: float
    usd: float
    total_samples: int = 3   # actual number of solver samples used (3 or 5)


# ── core logic ────────────────────────────────────────────────────────────────

def _solve_once(tracker: Tracker, question: str, run_id: str, tag_suffix: str) -> str | None:
    """Call the solver once and return the extracted answer (may be None)."""
    response = call_llm(
        tracker,
        model=SOLVER_MODEL,
        messages=[{"role": "user", "content": SOLVE_PROMPT.format(q=question)}],
        temperature=SOLVER_TEMP,
        run_id=run_id,
        tag=f"solve_{tag_suffix}",
    )
    return extract_model_answer(response)


def _compute_risk(answers: list[str | None]) -> tuple[str | None, float]:
    """
    Given a list of extracted answers, return:
      (modal_answer, risk_score)

    modal_answer: the most common non-None answer (None if all unparsed)
    risk_score:   1 - (modal_count / total_samples)  ∈ [0, 1]
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return None, 1.0  # all unparsed → maximum risk

    counts = Counter(valid)
    modal_answer, modal_count = counts.most_common(1)[0]
    risk = 1.0 - (modal_count / len(answers))
    return modal_answer, round(risk, 4)


def _audit(tracker: Tracker, question: str, answer: str | None,
           run_id: str, q_id: str) -> str | None:
    """Blind-solve audit (Option A):
    Step 1 — auditor solves independently without seeing the solver's answer.
    Step 2 — auditor sees the solver's answer and picks the more confident one.
    This avoids anchoring the auditor to the solver's (possibly wrong) answer.
    """
    # Step 1: blind solve
    blind_response = call_llm(
        tracker,
        model=AUDITOR_MODEL,
        messages=[{"role": "user", "content": AUDIT_PROMPT_BLIND_SOLVE.format(
            question=question,
        )}],
        temperature=AUDITOR_TEMP,
        run_id=run_id,
        tag=f"audit_blind_{q_id}",
    )
    auditor_answer = extract_model_answer(blind_response)

    # Step 2: compare with solver's answer and pick the better one
    solver_str  = str(answer) if answer is not None else "unable to determine"
    auditor_str = str(auditor_answer) if auditor_answer is not None else "unable to determine"

    compare_response = call_llm(
        tracker,
        model=AUDITOR_MODEL,
        messages=[{"role": "user", "content": AUDIT_PROMPT_COMPARE.format(
            auditor_answer=auditor_str,
            solver_answer=solver_str,
        )}],
        temperature=0.0,   # deterministic for the final pick
        run_id=run_id,
        tag=f"audit_compare_{q_id}",
    )
    return extract_model_answer(compare_response)


def run_one(item: dict, run_id: str) -> AuditResult:
    """Run the full adaptive-audit pipeline for a single question."""
    tracker = Tracker()
    q_id = item["id"]
    question = item["question"]
    gold = str(item["gold"])
    tag = item.get("tag", "unknown")

    # Sample 0 is the shared first answer (tag "solve"): the same cache entry
    # baseline, self-reflection, and adversarial use when run_id matches.
    # Later samples keep their own tags so the risk vote is not three copies.
    answers = []
    _text, first = solve_once(item, tracker, run_id)
    answers.append(first)
    for i in range(1, SAMPLE_K_INITIAL):
        ans = _solve_once(tracker, question, run_id=run_id, tag_suffix=f"{q_id}_s{i}")
        answers.append(ans)

    modal_answer, risk_score = _compute_risk(answers)

    # ── Step 3: DYNAMIC SAMPLING ──────────────────────────────────────────────
    # High risk (> RISK_HIGH): clearly uncertain, skip extra samples, go straight to audit
    # Low risk  (< RISK_LOW):  clearly confident, trust majority vote, stop here
    # Borderline (RISK_LOW <= risk <= RISK_HIGH): draw more samples and recompute
    if RISK_LOW <= risk_score <= RISK_HIGH:
        for i in range(SAMPLE_K_EXTRA):
            ans = _solve_once(tracker, question, run_id=run_id,
                              tag_suffix=f"{q_id}_extra{i}")
            answers.append(ans)
        modal_answer, risk_score = _compute_risk(answers)

    total_samples = len(answers)

    # ── Step 4: AUDIT (conditional) ───────────────────────────────────────────
    auditor_triggered = risk_score >= RISK_THRESHOLD
    auditor_answer = None
    if auditor_triggered:
        auditor_answer = _audit(tracker, question, modal_answer,
                                run_id=run_id, q_id=q_id)

    # ── Step 5: VERDICT ───────────────────────────────────────────────────────
    final_answer = auditor_answer if auditor_triggered else modal_answer
    correct = is_correct(final_answer, gold)
    majority_correct = is_correct(modal_answer, gold)

    return AuditResult(
        question_id=q_id,
        predicted=final_answer,
        gold=gold,
        correct=correct,
        tag=tag,
        risk_score=risk_score,
        auditor_triggered=auditor_triggered,
        majority_answer=modal_answer,
        majority_correct=majority_correct,
        auditor_answer=auditor_answer,
        calls=tracker.calls,
        in_tokens=tracker.in_tokens,
        out_tokens=tracker.out_tokens,
        latency_s=tracker.latency_s,
        usd=tracker.usd,
        total_samples=total_samples,
    )


# ── batch eval ────────────────────────────────────────────────────────────────

def run_eval(dataset: list[dict], run_id: str,
             workers: int = WORKERS) -> list[AuditResult]:
    """Run the adaptive auditor over a dataset in parallel."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, item, run_id): item for item in dataset}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            item = futures[fut]
            try:
                r = fut.result()
                results.append(r)
            except Exception as e:
                print(f"  ERROR on {item['id']}: {e}", file=sys.stderr)
            if i % 10 == 0 or i == len(dataset):
                print(f"  {i}/{len(dataset)} done", flush=True)
    return results


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(results: list[AuditResult]) -> dict:
    """Compute accuracy and cost metrics, broken down by tag."""
    total = len(results)
    if total == 0:
        return {}

    def _stats(subset):
        n = len(subset)
        if n == 0:
            return {"n": 0, "accuracy": None, "audit_rate": None,
                    "avg_calls": None, "avg_usd": None, "avg_risk": None}
        acc = sum(r.correct for r in subset) / n
        audit_rate = sum(r.auditor_triggered for r in subset) / n
        avg_calls = sum(r.calls for r in subset) / n
        avg_usd = sum(r.usd for r in subset) / n
        avg_risk = sum(r.risk_score for r in subset) / n
        return {
            "n": n,
            "accuracy": round(acc, 4),
            "audit_rate": round(audit_rate, 4),
            "avg_calls": round(avg_calls, 2),
            "avg_usd": round(avg_usd, 6),
            "avg_risk": round(avg_risk, 4),
        }

    by_tag = {}
    for tag in ("seeded_error", "consistent_correct", "always_wrong"):
        by_tag[tag] = _stats([r for r in results if r.tag == tag])

    # seeded-error audit precision: of questions the auditor fired on,
    # how many are actually seeded_error?  (tells us if the risk signal targets the right cases)
    audited = [r for r in results if r.auditor_triggered]
    if audited:
        precision = sum(r.tag == "seeded_error" for r in audited) / len(audited)
        recall_denom = sum(r.tag == "seeded_error" for r in results)
        recall = (sum(r.tag == "seeded_error" for r in audited) / recall_denom
                  if recall_denom else 0.0)
    else:
        precision = recall = 0.0

    # pre-audit vs post-audit: only meaningful for questions the auditor touched
    pre_post = {}
    if audited:
        for tag in ("seeded_error", "consistent_correct", "always_wrong"):
            subset = [r for r in audited if r.tag == tag]
            if not subset:
                pre_post[tag] = None
                continue
            n = len(subset)
            pre_acc  = sum(r.majority_correct for r in subset) / n
            post_acc = sum(r.correct for r in subset) / n
            improved = sum(not r.majority_correct and r.correct for r in subset)
            worsened = sum(r.majority_correct and not r.correct for r in subset)
            pre_post[tag] = {
                "n_audited": n,
                "pre_audit_acc":  round(pre_acc,  4),
                "post_audit_acc": round(post_acc, 4),
                "delta":          round(post_acc - pre_acc, 4),
                "improved":       improved,   # wrong → right
                "worsened":       worsened,   # right → wrong
            }
        # overall across all audited questions
        n = len(audited)
        pre_acc  = sum(r.majority_correct for r in audited) / n
        post_acc = sum(r.correct for r in audited) / n
        pre_post["overall_audited"] = {
            "n_audited": n,
            "pre_audit_acc":  round(pre_acc,  4),
            "post_audit_acc": round(post_acc, 4),
            "delta":          round(post_acc - pre_acc, 4),
            "improved":       sum(not r.majority_correct and r.correct for r in audited),
            "worsened":       sum(r.majority_correct and not r.correct for r in audited),
        }

    return {
        "overall": _stats(results),
        "by_tag": by_tag,
        "pre_vs_post_audit": pre_post,
        "auditor_targeting": {
            "precision_seeded": round(precision, 4),
            "recall_seeded": round(recall, 4),
        },
        "config": {
            "sample_k_initial": SAMPLE_K_INITIAL,
            "sample_k_extra": SAMPLE_K_EXTRA,
            "risk_low": RISK_LOW,
            "risk_high": RISK_HIGH,
            "risk_threshold": RISK_THRESHOLD,
            "solver_model": SOLVER_MODEL,
            "auditor_model": AUDITOR_MODEL,
        },
    }


def print_report(metrics: dict, run_id: str, results: list | None = None) -> None:
    """Print a human-readable summary to stdout."""
    ov = metrics["overall"]
    cfg = metrics["config"]
    print(f"\n{'='*60}")
    print(f"  Adaptive Auditor — run_id: {run_id}")
    print(f"  initial={cfg['sample_k_initial']}  extra={cfg['sample_k_extra']}  "
          f"risk_low={cfg['risk_low']}  risk_high={cfg['risk_high']}  "
          f"threshold={cfg['risk_threshold']}")
    print(f"  solver={cfg['solver_model']}  auditor={cfg['auditor_model']}")
    print(f"{'='*60}")
    avg_samples = sum(r.total_samples for r in results) / len(results) if results else 0
    print(f"  Overall    n={ov['n']:4d}  acc={ov['accuracy']:.1%}  "
          f"audit_rate={ov['audit_rate']:.1%}  "
          f"avg_samples={avg_samples:.1f}  avg_risk={ov['avg_risk']:.3f}")
    print(f"{'-'*60}")
    for tag, s in metrics["by_tag"].items():
        if s["n"] == 0:
            continue
        acc_str = f"{s['accuracy']:.1%}" if s["accuracy"] is not None else "  n/a"
        ar_str  = f"{s['audit_rate']:.1%}" if s["audit_rate"] is not None else "  n/a"
        print(f"  {tag:<22} n={s['n']:4d}  acc={acc_str}  audit_rate={ar_str}")
    tgt = metrics["auditor_targeting"]
    print(f"{'-'*60}")
    print(f"  Auditor targeting seeded_error:  "
          f"precision={tgt['precision_seeded']:.1%}  "
          f"recall={tgt['recall_seeded']:.1%}")
    print(f"{'='*60}")

    # pre vs post audit table
    pp = metrics.get("pre_vs_post_audit", {})
    if pp:
        print(f"\n  Pre-audit vs Post-audit  (audited questions only)")
        print(f"  {'tag':<22}  {'n':>4}  {'before':>7}  {'after':>7}  {'delta':>7}  {'↑improved':>9}  {'↓worsened':>9}")
        print(f"  {'-'*22}  {'-'*4}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*9}  {'-'*9}")
        for tag in ("seeded_error", "consistent_correct", "always_wrong", "overall_audited"):
            s = pp.get(tag)
            if s is None:
                continue
            label = "overall (audited)" if tag == "overall_audited" else tag
            delta_str = f"{s['delta']:+.1%}"
            print(f"  {label:<22}  {s['n_audited']:>4}  "
                  f"{s['pre_audit_acc']:>6.1%}  "
                  f"{s['post_audit_acc']:>6.1%}  "
                  f"{delta_str:>7}  "
                  f"{s['improved']:>9}  "
                  f"{s['worsened']:>9}")
        print()


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the Adaptive Auditor.")
    parser.add_argument("--split", choices=["dev", "test"], default="dev",
                        help="Which split to evaluate (default: dev)")
    parser.add_argument("--run-id", default=None,
                        help="Cache namespace; default shared_<split>, same first answer as the other systems")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--out", default=None,
                        help="Save per-question results to this JSON file")
    args = parser.parse_args()

    run_id = args.run_id or shared_run_id(args.split)

    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", f"{args.split}.json"
    )
    with open(data_path) as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} questions from {args.split}.json")
    print(f"run_id={run_id!r}  initial={SAMPLE_K_INITIAL}  extra={SAMPLE_K_EXTRA}  "
          f"risk_low={RISK_LOW}  risk_high={RISK_HIGH}  threshold={RISK_THRESHOLD}  "
          f"solver={SOLVER_MODEL}  auditor={AUDITOR_MODEL}")

    check_ready(SOLVER_MODEL)
    check_ready(AUDITOR_MODEL)

    results = run_eval(dataset, run_id=run_id, workers=args.workers)
    metrics = compute_metrics(results)
    print_report(metrics, run_id, results)

    # save metrics
    os.makedirs(RESULTS_DIR, exist_ok=True)
    metrics_path = os.path.join(RESULTS_DIR, f"{run_id}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {metrics_path}")

    # optionally save per-question results
    out_path = args.out or os.path.join(RESULTS_DIR, f"{run_id}_results.json")
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Per-question results → {out_path}")


if __name__ == "__main__":
    main()
