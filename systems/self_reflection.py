"""System 2: the same model answers, critiques that answer, then revises it.

The first call matches the baseline (SOLVE_PROMPT, SOLVER_TEMP, tag "solve"), so a shared
--run-id reuses that cached answer. Critique and revision are two further calls on the same model.
There is no separate auditor and no tools.

    python -m systems.self_reflection --split dev --run-id compare_dev
    python -m systems.self_reflection --split dev --run-id compare_dev --models llama3.2:3b --limit 20
    python -m systems.self_reflection --report --split dev --run-id compare_dev
    python -m systems.self_reflection --split test --run-id reflect_test_final --final
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (DATA_DIR, RESULTS_DIR, SEED, SOLVER_MODEL, SOLVER_TEMP, WORKERS,
                    shared_run_id)
from grading import extract_model_answer
from llm import Tracker, call_llm, check_ready
from prompts import SELF_CRITIQUE_PROMPT, SELF_REVISE_PROMPT, SOLVE_PROMPT
from systems.baseline import default_models, load_jsonl, model_slug, parse_models
from systems.common import finish, solve_once

VERSION = "self-reflection-v1"
TAGS = ("seeded_error", "consistent_correct", "always_wrong")


def fingerprint(model):
  """Changes when the prompts, temperature, revision policy, or model changes."""
  blob = json.dumps(
      [VERSION, SOLVE_PROMPT, SELF_CRITIQUE_PROMPT, SELF_REVISE_PROMPT, SOLVER_TEMP, model],
      sort_keys=True)
  return hashlib.sha256(blob.encode()).hexdigest()[:12]


def out_path(run_id, split, model):
  return f"{RESULTS_DIR}/{run_id}_{split}_self-reflection-{model_slug(model)}.jsonl"


def metrics_path(run_id, split, model):
  return f"{RESULTS_DIR}/{run_id}_{split}_self-reflection-{model_slug(model)}_metrics.json"


def _history(question, first_response, critique=None):
  """The conversation the model has already seen, plus an optional critique turn."""
  messages = [
      {"role": "user", "content": SOLVE_PROMPT.format(q=question)},
      {"role": "assistant", "content": first_response},
      {"role": "user", "content": SELF_CRITIQUE_PROMPT},
  ]
  if critique is not None:
    messages.append({"role": "assistant", "content": critique})
    messages.append({"role": "user", "content": SELF_REVISE_PROMPT})
  return messages


def run_self_reflection(item, run_id, model=None):
  """Solve, then critique, then revise. A failed revision keeps the first answer."""
  model = model or SOLVER_MODEL
  tr = Tracker()
  first_response, first = solve_once(item, tr, run_id, model=model)
  extra = {"model": model}

  critique = call_llm(
      tr, model=model, messages=_history(item["question"], first_response),
      temperature=SOLVER_TEMP, run_id=run_id, tag="self-critique")
  extra["critique"] = critique
  extra["critique_generated"] = bool(critique.strip())

  final, revised = first, False
  if extra["critique_generated"]:
    extra["revision_attempted"] = True
    revision = call_llm(
        tr, model=model,
        messages=_history(item["question"], first_response, critique),
        temperature=SOLVER_TEMP, run_id=run_id, tag="self-revise")
    revised_answer = extract_model_answer(revision)
    extra["revision_parsed"] = revised_answer is not None
    if revised_answer is not None:
      final, revised = revised_answer, True
    else:
      extra["revision_blocked"] = True
      extra["revision_failure"] = "revision had no parsable final answer"
  else:
    extra["revision_attempted"] = False
    extra["revision_parsed"] = False
    extra["revision_blocked"] = True
    extra["revision_failure"] = "critique was empty"
  extra.update(revised=revised, changed=(final != first))
  return finish(item, "self_reflection", tr, first, final, extra)


def _rate(rows, pred):
  n = len(rows)
  if n == 0:
    return None
  return round(sum(1 for r in rows if pred(r)) / n, 4)


def stats(rows):
  n = len(rows)
  if n == 0:
    return {"n": 0, "initial_accuracy": None, "final_accuracy": None,
            "initial_unparsed_rate": None, "final_unparsed_rate": None,
            "changed_rate": None, "fixed": 0, "broke": 0,
            "avg_calls": None, "avg_tokens": None, "avg_latency_s": None}
  fixed = sum(1 for r in rows if not r["initial_correct"] and r["final_correct"])
  broke = sum(1 for r in rows if r["initial_correct"] and not r["final_correct"])
  return {
      "n": n,
      "initial_accuracy": _rate(rows, lambda r: r["initial_correct"]),
      "final_accuracy": _rate(rows, lambda r: r["final_correct"]),
      "initial_unparsed_rate": _rate(rows, lambda r: r["initial_missing"]),
      "final_unparsed_rate": _rate(rows, lambda r: r["final_answer"] is None),
      "changed_rate": _rate(rows, lambda r: r.get("extra", {}).get("changed")),
      "fixed": fixed,
      "broke": broke,
      "avg_calls": round(sum(r["calls"] for r in rows) / n, 2),
      "avg_tokens": round(sum(r["in_tokens"] + r["out_tokens"] for r in rows) / n, 1),
      "avg_latency_s": round(sum(r["latency_s"] for r in rows) / n, 2),
  }


def compute_metrics(rows, model):
  by_tag = {}
  for tag in TAGS:
    by_tag[tag] = stats([r for r in rows if r.get("extra", {}).get("tag") == tag])
  return {
      "model": model,
      "overall": stats(rows),
      "by_tag": by_tag,
      "config": {"temperature": SOLVER_TEMP, "version": VERSION},
  }


def _fmt_pct(value):
  return "n/a" if value is None else f"{value:.1%}"


def _fmt_num(value):
  return "n/a" if value is None else f"{value:.1f}"


def print_report(metrics, run_id, split):
  ov = metrics["overall"]
  print(f"\n{'=' * 72}")
  print(f"  Self-reflection - {run_id} / {split} / {metrics['model']}")
  print(f"  temperature={metrics['config']['temperature']}  version={metrics['config']['version']}")
  print(f"{'=' * 72}")
  print(f"  Overall    n={ov['n']:4d}  first={_fmt_pct(ov['initial_accuracy'])}  "
        f"final={_fmt_pct(ov['final_accuracy'])}  "
        f"changed={_fmt_pct(ov['changed_rate'])}  "
        f"fixed={ov['fixed']}  broke={ov['broke']}")
  print(f"             unparsed first={_fmt_pct(ov['initial_unparsed_rate'])}  "
        f"final={_fmt_pct(ov['final_unparsed_rate'])}  "
        f"avg_calls={_fmt_num(ov['avg_calls'])}  "
        f"avg_tokens={_fmt_num(ov['avg_tokens'])}  "
        f"avg_latency_s={_fmt_num(ov['avg_latency_s'])}")
  print(f"{'-' * 72}")
  for tag, row in metrics["by_tag"].items():
    if row["n"] == 0:
      continue
    print(f"  {tag:<22} n={row['n']:4d}  first={_fmt_pct(row['initial_accuracy'])}  "
          f"final={_fmt_pct(row['final_accuracy'])}  "
          f"fixed={row['fixed']}  broke={row['broke']}")
  print(f"{'=' * 72}\n")


def print_comparison(all_metrics):
  if len(all_metrics) < 2:
    return
  print(f"\n{'=' * 78}")
  print("  Self-reflection comparison")
  print(f"  {'model':<22}{'n':>6}{'first':>10}{'final':>10}{'changed':>10}{'calls':>8}{'seconds':>10}")
  for metrics in all_metrics:
    ov = metrics["overall"]
    print(f"  {metrics['model']:<22}{ov['n']:6d}{_fmt_pct(ov['initial_accuracy']):>10}"
          f"{_fmt_pct(ov['final_accuracy']):>10}{_fmt_pct(ov['changed_rate']):>10}"
          f"{_fmt_num(ov['avg_calls']):>8}{_fmt_num(ov['avg_latency_s']):>10}")
  print(f"{'=' * 78}\n")


def save_metrics(path, metrics):
  os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
  print(f"Metrics saved -> {path}")


def run_one_model(items, run_id, split, model, workers):
  fp = fingerprint(model)
  path = out_path(run_id, split, model)
  os.makedirs(RESULTS_DIR, exist_ok=True)
  done = set()
  if os.path.exists(path):
    old = load_jsonl(path)
    if old and old[0].get("extra", {}).get("fingerprint") != fp:
      sys.exit(f"{path} was made with a different setup (prompt, temperature, or model changed).\n"
               f"Use a new --run-id so results are never mixed.")
    done = {r["qid"] for r in old}
  todo = [x for x in items if x["id"] not in done]
  print(f"model {model} | fingerprint {fp} | {len(todo)} to run, {len(done)} already done "
        f"| workers {workers}\n-> {path}")

  if todo:
    t0, finished, failed = time.time(), 0, 0
    with ThreadPoolExecutor(workers) as ex, open(path, "a", encoding="utf-8") as out:
      futs = {ex.submit(run_self_reflection, it, run_id, model): it for it in todo}
      for fut in as_completed(futs):
        it = futs[fut]
        try:
          result = fut.result()
        except Exception as e:
          failed += 1
          print(f"  ! {it['id']} failed ({e!r}); rerun to retry it")
          continue
        result.extra.update(fingerprint=fp, tag=it.get("tag"), source=it.get("source"),
                            level=it.get("level"))
        out.write(json.dumps(asdict(result)) + "\n")
        out.flush()
        finished += 1
        first = "missing" if result.initial_missing else ("right" if result.initial_correct else "WRONG")
        last = "missing" if result.final_answer is None else ("right" if result.final_correct else "WRONG")
        print(f"  {finished}/{len(todo)} {it['id']:<12} first={first:<7} final={last:<7} "
              f"answer={result.final_answer}")
    print(f"finished {finished}, failed {failed}, {time.time() - t0:.0f}s")

  rows = load_jsonl(path) if os.path.exists(path) else []
  metrics = compute_metrics(rows, model)
  print_report(metrics, run_id, split)
  save_metrics(metrics_path(run_id, split, model), metrics)
  print(f"Per-question results -> {path}")
  return metrics


def report_saved(run_id, split, models):
  found = []
  for model in models:
    path = out_path(run_id, split, model)
    if not os.path.exists(path):
      print(f"no result file for {model}: {path}")
      continue
    rows = load_jsonl(path)
    name = rows[0].get("extra", {}).get("model", model) if rows else model
    metrics = compute_metrics(rows, name)
    print_report(metrics, run_id, split)
    save_metrics(metrics_path(run_id, split, model), metrics)
    found.append(metrics)
  if not found:
    sys.exit(f"no result files for run-id {run_id} / {split}")
  print_comparison(found)


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--split", choices=["dev", "test"], default="dev")
  ap.add_argument("--run-id", default=None,
                  help="cache and result namespace; default shared_<split>, "
                       "which reuses the baseline solve call")
  ap.add_argument("--models", default=",".join(default_models()),
                  help="comma-separated Ollama models (default: solver, auditor, qwen2.5:7b)")
  ap.add_argument("--limit", type=int, default=None,
                  help="evaluate a prefix of the shuffled split")
  ap.add_argument("--workers", type=int, default=WORKERS)
  ap.add_argument("--final", action="store_true", help="required for the test split")
  ap.add_argument("--report", action="store_true",
                  help="print saved results for --models, no model calls")
  args = ap.parse_args()

  run_id = args.run_id or shared_run_id(args.split)
  models = parse_models(args.models)
  if args.report:
    report_saved(run_id, args.split, models)
    return
  if args.split == "test" and not args.final:
    sys.exit("The test split is sealed. Tune on dev; run test ONCE with --final.")
  if args.workers < 1:
    sys.exit("--workers must be at least 1")

  for model in models:
    check_ready(model)

  data_path = os.path.join(DATA_DIR, f"{args.split}.json")
  with open(data_path, encoding="utf-8") as f:
    items = json.load(f)
  random.Random(SEED).shuffle(items)
  if args.limit:
    items = items[:args.limit]
  print(f"Loaded {len(items)} questions from {args.split}.json")
  print(f"run_id={run_id!r}  models={models}  temperature={SOLVER_TEMP}")

  all_metrics = [run_one_model(items, run_id, args.split, model, args.workers) for model in models]
  print_comparison(all_metrics)


if __name__ == "__main__":
  main()
