#!/usr/bin/env python3
"""
Run the adversarial system over a split and report how well it works. Self-contained: needs no
other system, because the first answer IS the baseline answer (same cached call, same run_id).

    python -m scripts.eval_adversarial --run-id ev1 --limit 20          # first look (~17 min)
    python -m scripts.eval_adversarial --run-id ev1                     # whole dev split (resumes)
    python -m scripts.eval_adversarial --report --run-id ev1            # print the report, no model calls
    python -m scripts.eval_adversarial --inspect math_2924 gsm_383 --run-id ev1   # read specific transcripts
    python -m scripts.eval_adversarial --run-id ev1 --no-tools          # ablations: a separate file each
    python -m scripts.eval_adversarial --run-id ev1 --style neutral
    python -m scripts.eval_adversarial --run-id ev1 --auditor-model llama3.2:3b     # same model as solver
    python -m scripts.eval_adversarial --run-id final --split test --final          # ONCE, at the very end

RULES THAT KEEP THE NUMBERS HONEST
  * Use a NEW --run-id after ANY change to a prompt, the code, the model, or installed packages.
    Each result file is fingerprinted; resuming with a changed setup is refused, so runs never mix.
  * The test split is sealed: it needs --final. Tune on dev only.
  * Latency is only meaningful with --workers 1 (the default).
  * Questions run one at a time; a failed question is skipped and retried on the next run.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

from config import DATA_DIR, RESULTS_DIR, SEED, SOLVER_MODEL, SOLVER_TEMP
from grading import is_correct
from llm import check_ready
from prompts import AUDITOR_CRITIQUE_PROMPT, REVISE_PROMPT, SOLVE_PROMPT, build_auditor_system
from systems.adversarial import run_adversarial
from systems.auditor import AUDITOR_PIPELINE_VERSION, AuditorConfig
from tools import available_modules, is_echo_input


# ======================================================================= running
def make_cfg(a):
    if a.max_steps is not None and a.max_steps < 2:
        raise SystemExit("--max-steps must be at least 2 when tools are enabled")
    kw = {"use_tools": not a.no_tools, "style": a.style}
    if a.auditor_model:
        kw["model"] = a.auditor_model
    if a.max_steps:
        kw["max_steps"] = a.max_steps
    return AuditorConfig(**kw)


def fingerprint(cfg):
    """Changes whenever anything that affects behaviour changes (prompts, config, models, packages)."""
    blob = json.dumps([AUDITOR_PIPELINE_VERSION, SOLVE_PROMPT, AUDITOR_CRITIQUE_PROMPT, REVISE_PROMPT,
                       SOLVER_MODEL, SOLVER_TEMP, asdict(cfg),
                       build_auditor_system(cfg.style, cfg.use_tools, False, available_modules())],
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def out_path(run_id, split, cfg):
    return f"{RESULTS_DIR}/{run_id}_{split}_adversarial-{cfg.tag()}.jsonl"


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def run(a):
    cfg = make_cfg(a)
    if a.split == "test" and not a.final:
        sys.exit("The test split is sealed. Tune on dev; run test ONCE with --final.")
    check_ready(SOLVER_MODEL)
    check_ready(cfg.model)

    items = json.load(open(f"{DATA_DIR}/{a.split}.json"))
    if a.tag:
        items = [x for x in items if x.get("tag") == a.tag]
        print(f"NOTE: filtered to tag={a.tag}; this is NOT the natural distribution.")
    random.Random(SEED).shuffle(items)          # fixed order: --limit 20 is a prefix of --limit 40
    if a.limit:
        items = items[:a.limit]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path, fp = out_path(a.run_id, a.split, cfg), fingerprint(cfg)
    done = set()
    if os.path.exists(path):
        old = load(path)
        if old and old[0]["extra"].get("fingerprint") != fp:
            sys.exit(f"{path} was made with a different setup (prompt, code, model or packages changed).\n"
                     f"Use a new --run-id so results are never mixed.")
        done = {r["qid"] for r in old}
    todo = [x for x in items if x["id"] not in done]
    print(f"variant {cfg.tag()} | fingerprint {fp} | {len(todo)} to run, {len(done)} already done "
          f"| auditor {cfg.model} | workers {a.workers}\n-> {path}")
    if not todo:
        return report([path])

    t0, finished, failed = time.time(), 0, 0
    with ThreadPoolExecutor(a.workers) as ex, open(path, "a") as out:
        futs = {ex.submit(run_adversarial, it, a.run_id, cfg): it for it in todo}
        for f in as_completed(futs):
            it = futs[f]
            try:
                r = f.result()
            except Exception as e:
                failed += 1
                print(f"  ! {it['id']} failed ({e!r}); rerun to retry it")
                continue
            r.extra.update(fingerprint=fp, tag=it.get("tag"), source=it.get("source"), level=it.get("level"))
            out.write(json.dumps(asdict(r)) + "\n")
            out.flush()
            finished += 1
            el = time.time() - t0
            v = r.extra.get("verdict") or {}
            print(f"  {finished}/{len(todo)} {it['id']:<10} first={'right' if r.initial_correct else 'WRONG':<5} "
                  f"decision={v.get('decision', 'FAILED'):<10} final={'right' if r.final_correct else 'WRONG':<5} "
                  f"| {el/60:.1f}m elapsed, ETA {el/finished*(len(todo)-finished)/60:.1f}m", flush=True)
    if failed:
        print(f"\n{failed} questions failed; run the same command again to retry them.")
    report([path])


# ======================================================================= statistics
def bootstrap_ci(vals, n_boot=5000, seed=0):
    if not vals:
        return float("nan"), float("nan")
    rng, n = random.Random(seed), len(vals)
    means = sorted(sum(rng.choices(vals, k=n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def mcnemar_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def frac(a, b):
    return f"{a}/{b} = {100 * a / b:.1f}%" if b else f"{a}/0 = n/a"


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if p == p and r == r and p + r else float("nan")
    return p, r, f1


def fmt(x):
    return "n/a" if x != x else f"{100 * x:.1f}%"


# ======================================================================= report
def verdict(r):
    return r["extra"].get("verdict")


def gold_map():
    g = {}
    for sp in ("dev", "test"):
        p = f"{DATA_DIR}/{sp}.json"
        if os.path.exists(p):
            for x in json.load(open(p)):
                g[x["id"]] = x["gold"]
    return g


def show(name, rs):
    n = len(rs)
    print("\n" + "=" * 78 + f"\n{name}   (n={n})\n" + "=" * 78)
    if n < 40:
        print(f"NOTE: only {n} questions; every rate below has a wide margin of error.")

    first_ok = [int(r["initial_correct"]) for r in rs]
    final_ok = [int(r["final_correct"]) for r in rs]
    fixed = sum(1 for r in rs if not r["initial_correct"] and r["final_correct"])
    broke = sum(1 for r in rs if r["initial_correct"] and not r["final_correct"])
    supplied = sum(1 for r in rs if r["initial_missing"] and r["final_correct"])
    wrong0 = [r for r in rs if not r["initial_correct"]]
    right0 = [r for r in rs if r["initial_correct"]]
    lo, hi = bootstrap_ci([b - a for a, b in zip(first_ok, final_ok)])

    print("\n-- Accuracy (first answer = the baseline's answer)")
    print(f"   first answer: {100*sum(first_ok)/n:.1f}% correct  ->  after audit: {100*sum(final_ok)/n:.1f}% correct")
    print(f"   net change: {100*(sum(final_ok)-sum(first_ok))/n:+.1f} points  95% CI [{100*lo:+.1f}, {100*hi:+.1f}]   "
          f"(fixed {fixed}, broke {broke}; McNemar p={mcnemar_exact(fixed, broke):.3f})")
    print("\n-- Error catch / false correction")
    print(f"   error-catch rate (wrong -> right):       {frac(fixed, len(wrong0))}"
          f"   [of which {supplied} were missing answers supplied]")
    print(f"   false-correction rate (right -> wrong):  {frac(broke, len(right0))}")

    # decision x correctness
    cols = ("first right", "first wrong", "no answer")
    rows = ("agree", "disagree", "unverified", "model", "no_answer", "AUDITOR FAILED")
    tab = defaultdict(Counter)
    for r in rs:
        v = verdict(r)
        d = v["decision"] if v else "AUDITOR FAILED"
        c = cols[2] if r["initial_missing"] else (cols[0] if r["initial_correct"] else cols[1])
        tab[d][c] += 1
    print("\n-- What the auditor decided, by whether the first answer was actually right")
    print(f"   {'decision':<16}" + "".join(f"{c:>13}" for c in cols))
    for d in rows:
        if d in tab:
            print(f"   {d:<16}" + "".join(f"{tab[d][c]:>13}" for c in cols))

    # detectors
    answered = [r for r in rs if not r["initial_missing"]]
    def detector(flag):
        tp = sum(1 for r in answered if flag(r) and not r["initial_correct"])
        fp = sum(1 for r in answered if flag(r) and r["initial_correct"])
        fn = sum(1 for r in answered if not flag(r) and not r["initial_correct"])
        return prf(tp, fp, fn), sum(1 for r in answered if flag(r)), tp, fp, fn
    print("\n-- Auditor as an error DETECTOR (positive = 'the first answer is wrong'; answered questions only)")
    uses_grounded = any(verdict(r) and "grounded" in verdict(r) for r in rs)
    dets = [
        ("high-confidence solution", lambda r: bool(verdict(r))
         and verdict(r)["decision"] == "disagree"),
        ("auditor answer differs", lambda r: bool(verdict(r))
         and verdict(r).get("candidate_disagreement", False)),
    ]
    print(f"   {'detector':<24}{'flagged':>9}{'TP':>5}{'FP':>5}{'FN':>5}{'precision':>11}{'recall':>9}{'F1':>8}")
    for nm, fl in dets:
        (p, rc, f1), nf, tp, fp, fn = detector(fl)
        print(f"   {nm:<24}{nf:>9}{tp:>5}{fp:>5}{fn:>5}{fmt(p):>11}{fmt(rc):>9}{fmt(f1):>8}")
    base = 1 - sum(r["initial_correct"] for r in answered) / max(len(answered), 1)
    print(f"   (a detector that flagged everything would have precision {fmt(base)})")

    # revision behaviour
    rev = [r for r in rs if r["extra"].get("revised")]
    attempted = [r for r in rs if r["extra"].get("revision_attempted")]
    critiques = [r for r in rs if r["extra"].get("critique_generated")]
    blocked = [r for r in rs if r["extra"].get("revision_blocked")]
    critique_workflow = any("critique_generated" in r["extra"] for r in rs)
    print("\n-- Critique and revision")
    if critique_workflow:
        print(f"   auditor critiques generated: {frac(len(critiques), n)}")
        print(f"   solver retries attempted: {frac(len(attempted), n)}")
        print(f"   retries with a parsed final answer: "
              f"{frac(sum(1 for r in attempted if r['extra'].get('revision_parsed')), len(attempted))}")
        print(f"   retries not applied (missing critique, failed audit, or unparsable answer): "
              f"{frac(len(blocked), n)}")
    else:
        print("   this result file predates the critique-and-retry workflow; those metrics were not recorded")
    print(f"   final answers changed from first attempt: "
          f"{frac(sum(1 for r in rev if r['extra'].get('changed')), n)}")
    for lab, grp in (("first answer wrong", [r for r in rev if not r["initial_correct"]]),
                     ("first answer right", [r for r in rev if r["initial_correct"]])):
        ch = sum(1 for r in grp if r["extra"].get("changed"))
        print(f"   {lab}: {len(grp)} revised, solver changed its answer in {ch}; "
              f"now right in {sum(r['final_correct'] for r in grp)}")

    # auditor behaviour
    aud = [r for r in rs if verdict(r) and not r["extra"].get("audit_skipped")]
    if aud:
        tools = [len(r["extra"]["tool_calls"]) for r in aud]
        calls = [tc for r in aud for tc in r["extra"]["tool_calls"]]
        errs = sum(1 for tc in calls if tc["output"].startswith("ERROR"))
        print("\n-- Auditor behaviour")
        print(f"   avg steps {sum(r['extra']['steps'] for r in aud)/len(aud):.1f}, avg tool calls {sum(tools)/len(aud):.1f}, "
              f"tool calls that errored {frac(errs, len(calls))}")
        print(f"   audits with no tool call at all: {frac(sum(1 for t in tools if t == 0), len(aud))}")
        if uses_grounded:
            print(f"   high-confidence independent solutions: "
                  f"{frac(sum(1 for r in aud if verdict(r)['grounded']), len(aud))}")
            print(f"   solutions with matching successful tool output: "
                  f"{frac(sum(1 for r in aud if verdict(r).get('grounding_checks', 0) > 0), len(aud))}")
            print(f"   solutions with matching outputs from both tools: "
                  f"{frac(sum(1 for r in aud if verdict(r).get('method_diverse')), len(aud))}")
            print(f"   audits rejected for approximate evidence: "
                  f"{frac(sum(1 for r in aud if verdict(r).get('approximate_evidence')), len(aud))}")
            print(f"   audits with incomplete claim evidence: "
                  f"{frac(sum(1 for r in aud if not verdict(r).get('evidence_backed')), len(aud))}")
            not_eligible = sum(
                1 for r in aud
                if verdict(r).get("candidate_disagreement")
                and not verdict(r).get("revision_eligible"))
            print(f"   differing answers not yet revision-eligible: "
                  f"{frac(not_eligible, len(aud))}")
    fails = [r for r in rs if r["extra"].get("failure")]
    safe_stops = [r for r in fails if r["extra"].get("safe_stop")]
    invalid_fails = [r for r in fails if not r["extra"].get("safe_stop")]
    print(f"   auditor failures (invalid JSON etc.): {frac(len(invalid_fails), n)}")
    print(f"   safe stops after repeated malformed tool calls: {frac(len(safe_stops), n)}")
    candidates = [r for r in rs if critique_workflow and verdict(r)
                  and verdict(r).get("candidate_disagreement")
                  and r["extra"].get("revision_blocked")]
    if candidates:
        print(f"   differing answers not retried: {len(candidates)}")
        for r in candidates:
            v = verdict(r)
            print(f"      {r['qid']}: auditor={v.get('auditor_answer', '')!r}, "
                  f"checks={v.get('grounding_checks', 0)}, "
                  f"status={v.get('grounding_status', 'unknown')}, "
                  f"revision failure={r['extra'].get('revision_failure', 'unknown')}")

    gold = gold_map()
    ga = [r for r in rs if verdict(r) and verdict(r).get("grounded") and r["qid"] in gold]
    if ga:
        ar = sum(1 for r in ga if is_correct(verdict(r)["auditor_answer"], gold[r["qid"]]))
        print("\n-- The auditor's own explicitly marked tool answers, judged against the gold answer")
        print(f"   on {len(ga)} audits: auditor right {frac(ar, len(ga))}; "
              f"the solver's first answer on the same questions: {frac(sum(r['initial_correct'] for r in ga), len(ga))}")
        dis = [r for r in ga if verdict(r)["decision"] == "disagree" and not r["initial_correct"]]
        if dis:
            print(f"   when it disagreed with a WRONG first answer (n={len(dis)}): its own answer was right in "
                  f"{sum(1 for r in dis if is_correct(verdict(r)['auditor_answer'], gold[r['qid']]))}, "
                  f"and the solver ended right in {sum(r['final_correct'] for r in dis)}")
        agr = [r for r in ga if verdict(r)["decision"] == "agree" and not r["initial_correct"]]
        if agr:
            print(f"   it AGREED with a wrong first answer {len(agr)} times (a shared blind spot, or an echo: see below)")
    if aud:
        errs = Counter(tc["output"][:75].replace("\n", " ") for r in aud
                       for tc in r["extra"]["tool_calls"] if tc["output"].startswith("ERROR"))
        if errs:
            print("\n-- Most common tool errors")
            for msg, k in errs.most_common(5):
                print(f"   {k:>3}x  {msg}")
        echo_calls = [(r, tc) for r in aud
                      for tc in r["extra"]["tool_calls"]
                      if tc.get("echo") or is_echo_input(tc["tool"], tc["input"], r["initial_answer"])]
        print(f"\n-- Echo check: tool calls that merely print the solver's own answer: {len(echo_calls)} "
              f"(in {len({r['qid'] for r, _ in echo_calls})} audits)")
        independent = [verdict(r).get("grounding_checks", 0) for r in aud]
        print(f"   average matching checks: {sum(independent) / len(independent):.1f}")

    # cost
    print("\n-- Cost per question (means)         calls    tokens   seconds")
    per_call_metrics = all(
        all("in_tokens" in entry and "out_tokens" in entry and "latency_s" in entry
            for entry in r["log"])
        for r in rs)
    if per_call_metrics:
        def part(r, solve):
            ent = [e for e in r["log"] if (e["tag"] == "solve") == solve]
            return (len(ent), sum(e["in_tokens"] + e["out_tokens"] for e in ent),
                    sum(e["latency_s"] for e in ent))

        baseline = [part(r, True) for r in rs]
        extra = [part(r, False) for r in rs]
        mean = lambda values, index: sum(value[index] for value in values) / n
        print(f"   first answer (= the baseline)   {mean(baseline, 0):7.1f} "
              f"{mean(baseline, 1):9.0f} {mean(baseline, 2):9.1f}")
        print(f"   audit + revision (extra)        {mean(extra, 0):7.1f} "
              f"{mean(extra, 1):9.0f} {mean(extra, 2):9.1f}")
        ratio = ((mean(baseline, 2) + mean(extra, 2)) / mean(baseline, 2)
                 if mean(baseline, 2) else float("nan"))
        print(f"   -> the adversarial system costs {ratio:.1f}x the baseline in time")
    else:
        baseline_calls = sum(
            sum(entry.get("tag") == "solve" for entry in r["log"]) for r in rs) / n
        extra_calls = sum(
            sum(entry.get("tag") != "solve" for entry in r["log"]) for r in rs) / n
        total_tokens = sum(r["in_tokens"] + r["out_tokens"] for r in rs) / n
        total_seconds = sum(r["latency_s"] for r in rs) / n
        print(f"   first answer (= the baseline)   {baseline_calls:7.1f}       n/a       n/a")
        print(f"   audit + revision (extra)        {extra_calls:7.1f}       n/a       n/a")
        print(f"   total system                    {baseline_calls + extra_calls:7.1f} "
              f"{total_tokens:9.0f} {total_seconds:9.1f}")
        print("   per-call token and latency data is absent from the legacy LLM log; "
              "only total system cost is available")
    net = fixed - broke
    print("\n-- Is it worth paying for? (on this sample)")
    if per_call_metrics:
        extra_s = sum(part(r, False)[2] for r in rs)
        print(f"   net fixes {net:+d} over {n} questions = {100*net/n:+.1f} per 100 questions; "
              f"extra compute {extra_s/60:.1f} min")
    else:
        print(f"   net fixes {net:+d} over {n} questions = {100*net/n:+.1f} per 100 questions; "
              "extra compute unavailable from legacy per-call logs")
    if net > 0 and per_call_metrics:
        print(f"   {extra_s/net:.0f} extra seconds per net fix: auditing pays off only if one wrong answer is worth more than that")
    elif net <= 0:
        print("   no net improvement: auditing does not pay for itself on this sample")

    # breakdowns
    def breakdown(title, keyf):
        g = defaultdict(list)
        for r in rs:
            g[keyf(r)].append(r)
        print(f"\n-- By {title}\n   {'':<22}{'n':>4}{'first':>8}{'final':>8}{'fixed':>7}{'broke':>7}")
        for k in sorted(g, key=str):
            x = g[k]
            print(f"   {str(k):<22}{len(x):>4}{100*sum(r['initial_correct'] for r in x)/len(x):>7.0f}%"
                  f"{100*sum(r['final_correct'] for r in x)/len(x):>7.0f}%"
                  f"{sum(1 for r in x if not r['initial_correct'] and r['final_correct']):>7}"
                  f"{sum(1 for r in x if r['initial_correct'] and not r['final_correct']):>7}")
    breakdown("pilot tag (seeded_error rows are selected for failing once: report them separately)",
              lambda r: r["extra"].get("tag"))
    breakdown("source / level", lambda r: (r["extra"].get("source"), r["extra"].get("level")))


def inspect_ids(paths, ids):
    gold = gold_map()
    for p in paths:
        for r in load(p):
            if r["qid"] not in ids:
                continue
            v = verdict(r) or {}
            q = r["log"][0]["messages"][0]["content"].split("Problem:\n", 1)[-1].split("\n\nFinish with")[0]
            print("=" * 78 + f"\n{r['qid']}   [{os.path.basename(p)}]   gold={gold.get(r['qid'])}")
            print("   " + " ".join(q.split())[:300])
            print(f"   first answer {r['initial_answer']!r} ({'right' if r['initial_correct'] else 'WRONG'})"
                  f" -> final {r['final_answer']!r} ({'right' if r['final_correct'] else 'WRONG'})")
            print(f"   decision={v.get('decision')}  high_confidence={v.get('grounded')}  "
                  f"auditor_solution_status={v.get('model_overall')}  "
                  f"auditor_answer={v.get('auditor_answer')!r}  target={v.get('target')!r}")
            for i, tc in enumerate(r["extra"].get("tool_calls", []), 1):
                note = "   <-- just echoes the solver's answer" if (
                    tc.get("echo") or is_echo_input(tc["tool"], tc["input"], r["initial_answer"])
                ) else ""
                print(f"   tool {i} ({tc['tool']}){note}")
                print("      input : " + tc["input"].replace("\n", "\n              "))
                print("      output: " + tc["output"][:300].replace("\n", "\n              "))
            for c in v.get("claims", []):
                print(f"   [{c['status']}] {c['text'][:80]} :: {c['evidence'][:120]}")


def report(paths):
    data = {}
    for p in paths:
        rs = load(p)
        if rs:
            data[os.path.basename(p).replace(".jsonl", "")] = rs
            show(os.path.basename(p).replace(".jsonl", ""), rs)
    names = list(data)
    if len(names) > 1:
        print("\n" + "=" * 78 + "\nVARIANTS COMPARED on shared questions (final-answer correctness, exact McNemar)\n" + "=" * 78)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                A = {r["qid"]: r for r in data[names[i]]}
                B = {r["qid"]: r for r in data[names[j]]}
                common = set(A) & set(B)
                b = sum(1 for q in common if A[q]["final_correct"] and not B[q]["final_correct"])
                c = sum(1 for q in common if not A[q]["final_correct"] and B[q]["final_correct"])
                print(f"   {names[i]}  vs  {names[j]}: n={len(common)}  only-first-right={b}  only-second-right={c}  "
                      f"p={mcnemar_exact(b, c):.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id", default="ev1")
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default=None, choices=["seeded_error", "consistent_correct", "always_wrong"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--final", action="store_true", help="required for the test split")
    ap.add_argument("--report", action="store_true", help="only print the report for existing files")
    ap.add_argument("--inspect", nargs="+", metavar="QID",
                    help="print the auditor's tool calls, verdict and outcome for these questions")
    ap.add_argument("--auditor-model", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--no-tools", action="store_true")
    ap.add_argument("--style", choices=["adversarial", "neutral"], default="adversarial")
    a = ap.parse_args()
    if a.inspect:
        paths = sorted(glob.glob(f"{RESULTS_DIR}/{a.run_id}_{a.split}_adversarial-*.jsonl"))
        return inspect_ids(paths, set(a.inspect)) if paths else sys.exit("no result files found")
    if a.report:
        paths = sorted(glob.glob(f"{RESULTS_DIR}/{a.run_id}_{a.split}_adversarial-*.jsonl"))
        return report(paths) if paths else sys.exit(f"no result files for run-id {a.run_id} / {a.split}")
    run(a)


if __name__ == "__main__":
    main()