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

from config import RESULTS_DIR, SOLVER_MODEL, SOLVER_TEMP, WORKERS
from grading import extract_model_answer, is_correct
from llm import Tracker, call_llm, check_ready
from prompts import SOLVE_PROMPT

# ── tuning knobs ─────────────────────────────────────────────────────────────
SAMPLE_K = 5          # total solver samples per question (including the first)
RISK_THRESHOLD = 0.4  # audit when risk >= this value; tune on dev.json only

AUDIT_PROMPT = (
    "You are a strict math auditor. A student has attempted the following problem.\n\n"
    "Problem:\n{question}\n\n"
    "Student's answer: {answer}\n\n"
    "Check the answer carefully:\n"
    "  1. Identify the key steps needed to solve the problem.\n"
    "  2. State whether the student's answer is correct or incorrect.\n"
    "  3. If incorrect, work out the right answer.\n\n"
    "Finish with a single line in exactly this form, with nothing after it:\n"
    "ANSWER: <correct final answer as a number or simple fraction, no units>"
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
    risk_score: float        # 0.0 = unanimous, 1.0 = maximally split
    auditor_triggered: bool  # did we pay for a second call?
    auditor_answer: str | None  # what the auditor said (None if not triggered)

    # cost metadata
    calls: int
    in_tokens: int
    out_tokens: int
    latency_s: float
    usd: float


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
    """Ask the auditor to verify (or correct) the given answer."""
    answer_str = str(answer) if answer is not None else "unable to determine"
    response = call_llm(
        tracker,
        model=SOLVER_MODEL,
        messages=[{"role": "user", "content": AUDIT_PROMPT.format(
            question=question,
            answer=answer_str,
        )}],
        temperature=0.0,   # auditor is deterministic
        run_id=run_id,
        tag=f"audit_{q_id}",
    )
    return extract_model_answer(response)


def run_one(item: dict, run_id: str) -> AuditResult:
    """Run the full adaptive-audit pipeline for a single question."""
    tracker = Tracker()
    q_id = item["id"]
    question = item["question"]
    gold = str(item["gold"])
    tag = item.get("tag", "unknown")

    # ── Step 1+2: SOLVE × SAMPLE_K ───────────────────────────────────────────
    answers = []
    for i in range(SAMPLE_K):
        ans = _solve_once(tracker, question, run_id=run_id, tag_suffix=f"{q_id}_s{i}")
        answers.append(ans)

    # ── Step 3: RISK ──────────────────────────────────────────────────────────
    modal_answer, risk_score = _compute_risk(answers)

    # ── Step 4: AUDIT (conditional) ───────────────────────────────────────────
    auditor_triggered = risk_score >= RISK_THRESHOLD
    auditor_answer = None
    if auditor_triggered:
        auditor_answer = _audit(tracker, question, modal_answer,
                                run_id=run_id, q_id=q_id)

    # ── Step 5: VERDICT ───────────────────────────────────────────────────────
    final_answer = auditor_answer if auditor_triggered else modal_answer
    correct = is_correct(final_answer, gold)

    return AuditResult(
        question_id=q_id,
        predicted=final_answer,
        gold=gold,
        correct=correct,
        tag=tag,
        risk_score=risk_score,
        auditor_triggered=auditor_triggered,
        auditor_answer=auditor_answer,
        calls=tracker.calls,
        in_tokens=tracker.in_tokens,
        out_tokens=tracker.out_tokens,
        latency_s=tracker.latency_s,
        usd=tracker.usd,
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

    return {
        "overall": _stats(results),
        "by_tag": by_tag,
        "auditor_targeting": {
            "precision_seeded": round(precision, 4),
            "recall_seeded": round(recall, 4),
        },
        "config": {
            "sample_k": SAMPLE_K,
            "risk_threshold": RISK_THRESHOLD,
            "model": SOLVER_MODEL,
        },
    }


def print_report(metrics: dict, run_id: str) -> None:
    """Print a human-readable summary to stdout."""
    ov = metrics["overall"]
    print(f"\n{'='*60}")
    print(f"  Adaptive Auditor — run_id: {run_id}")
    print(f"  SAMPLE_K={metrics['config']['sample_k']}  "
          f"RISK_THRESHOLD={metrics['config']['risk_threshold']}  "
          f"model={metrics['config']['model']}")
    print(f"{'='*60}")
    print(f"  Overall    n={ov['n']:4d}  acc={ov['accuracy']:.1%}  "
          f"audit_rate={ov['audit_rate']:.1%}  "
          f"avg_calls={ov['avg_calls']:.1f}  avg_risk={ov['avg_risk']:.3f}")
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
    print(f"{'='*60}\n")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run the Adaptive Auditor.")
    parser.add_argument("--split", choices=["dev", "test"], default="dev",
                        help="Which split to evaluate (default: dev)")
    parser.add_argument("--run-id", default=None,
                        help="Cache namespace; defaults to 'adaptive_<split>'")
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--out", default=None,
                        help="Save per-question results to this JSON file")
    args = parser.parse_args()

    run_id = args.run_id or f"adaptive_{args.split}"

    data_path = os.path.join(
        os.path.dirname(__file__), "..", "data", f"{args.split}.json"
    )
    with open(data_path) as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} questions from {args.split}.json")
    print(f"run_id={run_id!r}  SAMPLE_K={SAMPLE_K}  "
          f"RISK_THRESHOLD={RISK_THRESHOLD}  model={SOLVER_MODEL}")

    check_ready(SOLVER_MODEL)

    results = run_eval(dataset, run_id=run_id, workers=args.workers)
    metrics = compute_metrics(results)
    print_report(metrics, run_id)

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
