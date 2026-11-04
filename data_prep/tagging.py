"""Pilot tagging: sample the solver k times per question and label each question
by how reliably it is solved.

    5/5 correct   -> consistent_correct   (used for false-correction rate)
    1-4/5 correct -> seeded_error         (used for error-catch rate)
    0/5 correct   -> always_wrong         (review by hand: bad gold? ambiguous? too hard?)

IMPORTANT: these runs are for LABELLING ONLY. Never score a system on them
(selection + regression to the mean would inflate any later improvement).
"""
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (DATA_DIR, PILOT_K, PILOT_TEMPERATURE, SEED, SOLVER_MODEL, WORKERS)
from data_prep.splits import load_all, save_all
from grading import extract_with_mode, is_correct, normalize, to_fraction
from llm import Tracker, call_llm
from prompts import SOLVE_PROMPT

PILOT_RUN_ID = "pilot"


def tag_from_rate(rate):
  if rate == 1.0:
    return "consistent_correct"
  if rate == 0.0:
    return "always_wrong"
  return "seeded_error"


def _one_sample(item, k, model, temperature):
  """One solver attempt. The sample index k goes into the cache tag, so the
  k attempts at the same prompt are genuinely separate calls."""
  tr = Tracker()
  text = call_llm(tr, model=model, temperature=temperature, run_id=PILOT_RUN_ID,
                  tag=f"k{k}",
                  messages=[{"role": "user",
                              "content": SOLVE_PROMPT.format(q=item["question"])}])
  pred, mode = extract_with_mode(text)
  return item["id"], k, pred, is_correct(pred, item["gold"]), tr, mode


def run_pilot(limit=None, k=PILOT_K, model=SOLVER_MODEL,
              temperature=PILOT_TEMPERATURE, workers=WORKERS, seed=SEED,
              save_every=10):
  """Tag questions in data/all.json (all of them, or the first `limit` of a fixed shuffle).

  * Subsets are nested: --limit 10 is contained in --limit 50, so already-cached calls are reused.
  * Progress is printed live with an ETA, and tags are saved to disk every `save_every`
    questions (and on Ctrl-C), so an interrupted run loses nothing.
  * A call that fails (e.g. timeout) is reported and skipped; that question stays untagged.
  """
  items = load_all()
  order = items[:]
  random.Random(seed).shuffle(order)
  todo = order[:limit] if limit else order
  by_id = {it["id"]: it for it in items}

  jobs = [(it, j) for it in todo for j in range(k)]
  print(f"{len(jobs)} calls = {len(todo)} questions x k={k} on {model} "
        f"(temp {temperature}, workers {workers})", flush=True)

  preds = defaultdict(dict)            # qid -> {k: pred}
  oks = defaultdict(dict)              # qid -> {k: bool}
  modes = defaultdict(dict)            # qid -> {k: how the answer was parsed}
  failures = []
  total = Tracker()
  state = {"done_q": 0, "unsaved": 0}
  t_start = time.time()

  def finish_question(qid):
    flags = [oks[qid][j] for j in sorted(oks[qid])]
    rate = sum(flags) / len(flags)
    by_id[qid].update({
      "pilot_pass_rate": rate,
      "pilot_preds": [preds[qid][j] for j in sorted(preds[qid])],
      "pilot_unparsed": sum(1 for j in preds[qid] if preds[qid][j] is None),
      "pilot_fallback": sum(1 for j in modes[qid] if modes[qid][j] == "last_line"),
      "tag": tag_from_rate(rate)})
    state["done_q"] += 1
    state["unsaved"] += 1
    if state["unsaved"] >= save_every:
      save_all(items)
      state["unsaved"] = 0

  ex = ThreadPoolExecutor(workers)
  futs = {ex.submit(_one_sample, it, j, model, temperature): (it["id"], j) for it, j in jobs}
  try:
    for n, f in enumerate(as_completed(futs), 1):
        qid, j = futs[f]
        try:
          _, _, pred, ok, tr, mode = f.result()
        except Exception as e:
          failures.append((qid, j, repr(e)))
          print(f"  ! call failed for {qid} (sample {j}): {e!r}", flush=True)
          continue
        preds[qid][j], oks[qid][j], modes[qid][j] = pred, ok, mode
        total.calls += tr.calls
        total.in_tokens += tr.in_tokens
        total.out_tokens += tr.out_tokens
        total.usd += tr.usd
        if len(oks[qid]) == k:
          finish_question(qid)
        if n <= 3 or n % 10 == 0 or n == len(jobs):
          el = time.time() - t_start
          eta = el / n * (len(jobs) - n)
          print(f"  {n}/{len(jobs)} calls | {state['done_q']}/{len(todo)} questions tagged | "
                f"last call {tr.latency_s:.1f}s | elapsed {el/60:.1f}m | ETA {eta/60:.1f}m",
                flush=True)
  except KeyboardInterrupt:
    print("\nInterrupted: saving what is done so far...", flush=True)
    ex.shutdown(wait=False, cancel_futures=True)
  finally:
    ex.shutdown(wait=False, cancel_futures=True)
    save_all(items)
    write_review_file(items)

  cost = f"~${total.usd:.2f} at the PRICES in config.py" if total.usd else "no price set / local model"
  print(f"\npilot tokens: {total.in_tokens:,} in / {total.out_tokens:,} out ({cost})")
  if failures:
    print(f"{len(failures)} calls failed; those questions are untagged. Rerun the same command "
          f"to retry them (finished calls are cached).")
  return items


def _answer_key(a):
  f = to_fraction(a)
  return str(f) if f is not None else normalize(a)


def write_review_file(items):
  """Always-wrong questions with the gold answer next to what the solver said.

  `suspect_label` is True when the solver gave the SAME answer 3+ times and it disagrees with
  the gold: the classic sign of a wrong gold label or a question/answer mismatch. Those are
  listed first. Scattered wrong answers usually just mean the question is hard."""
  review = []
  for it in items:
    if it.get("tag") != "always_wrong":
      continue
    keys = [_answer_key(p) for p in (it.get("pilot_preds") or []) if p is not None]
    top, n = Counter(keys).most_common(1)[0] if keys else (None, 0)
    review.append({"id": it["id"], "source": it["source"], "level": it["level"],
                    "suspect_label": n >= 3, "most_common_answer": top,
                    "most_common_count": n, "gold": it["gold"],
                    "solver_answers": it.get("pilot_preds"), "question": it["question"]})
  review.sort(key=lambda r: (not r["suspect_label"], r["id"]))
  json.dump(review, open(f"{DATA_DIR}/review_always_wrong.json", "w"), indent=2)
  return review


def summary(items=None):
  items = items if items is not None else load_all()
  tagged = [x for x in items if "pilot_pass_rate" in x]
  print(f"\n== dataset: {len(items)} questions, {len(tagged)} tagged ==")
  if not tagged:
    print("no pilot results yet")
    return

  acc = sum(x["pilot_pass_rate"] for x in tagged) / len(tagged)
  print(f"mean per-sample accuracy: {acc:.1%}   (target roughly 55-75%)")
  print("tags:", dict(Counter(x["tag"] for x in tagged)))
  print("tags by split:", dict(Counter((x["split"], x["tag"]) for x in tagged)))

  print("\naccuracy by source/level:")
  groups = defaultdict(list)
  for x in tagged:
    groups[(x["source"], x["level"])].append(x["pilot_pass_rate"])
  for key in sorted(groups, key=str):
    v = groups[key]
    print(f"  {str(key):<18} {sum(v)/len(v):>6.1%}   (n={len(v)})")

  n_samples = sum(len(x["pilot_preds"]) for x in tagged)
  n_unparsed = sum(x["pilot_unparsed"] for x in tagged)
  n_fb = sum(x.get("pilot_fallback", 0) for x in tagged)
  print(f"\nunparsable answers: {n_unparsed}/{n_samples} samples ({n_unparsed/n_samples:.1%})")
  if n_unparsed / n_samples > 0.03:
    print("  -> HIGH: counted as wrong. Run: python -m scripts.diagnose_unparsed")
  print(f"read from the final line (no ANSWER marker): {n_fb}/{n_samples} samples ({n_fb/n_samples:.1%})"
        "   <- report this in your methods")
  if n_fb / n_samples > 0.25:
    print("  -> the solver often ignores the output format. The fallback copes, but a one-line "
          "worked example in the prompt would make answers cleaner.")

  if acc > 0.80:
    print("\n-> accuracy too high: few errors to study. Use MATH levels 4-5 only or a weaker solver.")
  elif acc < 0.40:
    print("\n-> accuracy too low: use easier levels or a stronger solver.")
  n_seeded = sum(1 for x in tagged if x["tag"] == "seeded_error")
  if n_seeded < 60:
    print(f"-> only {n_seeded} seeded_error questions; error-catch estimates will be noisy "
          f"(aim for 60+, ideally 100+).")


def apply_exclusions(ids, reason="bad gold label / ambiguous (manual review)"):
  """Remove reviewed-and-rejected questions from all/dev/test and log why."""
  items = load_all()
  ids = set(ids)
  kept = [x for x in items if x["id"] not in ids]
  dropped = [x["id"] for x in items if x["id"] in ids]
  unknown = ids - set(dropped)
  save_all(kept)

  log_path = f"{DATA_DIR}/removed_log.json"
  try:
    log = json.load(open(log_path))
  except FileNotFoundError:
    log = []
  log += [{"id": i, "reason": reason} for i in dropped]
  json.dump(log, open(log_path, "w"), indent=2)
  write_review_file(kept)
  print(f"removed {len(dropped)} questions; {len(kept)} remain.")
  if unknown:
    print(f"not found (ignored): {sorted(unknown)}")