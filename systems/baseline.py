"""System 1: one LLM call that returns the solver's first answer.

The call uses SOLVE_PROMPT, SOLVER_TEMP, and tag "solve". On config.SOLVER_MODEL that is the
same cached first answer the other systems start from when they share a --run-id.

    ollama pull llama3.2:3b
    ollama pull llama3.1:8b
    ollama pull qwen2.5:7b
    python -m systems.baseline --split dev --run-id baseline_dev
    python -m systems.baseline --split dev --run-id baseline_dev --models llama3.2:3b --limit 20
    python -m systems.baseline --report --split dev --run-id baseline_dev
    python -m systems.baseline --split test --run-id baseline_test_final --final
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import (AUDITOR_MODEL, DATA_DIR, RESULTS_DIR, SEED, SOLVER_MODEL,
                    SOLVER_TEMP, WORKERS)
from llm import Tracker, check_ready
from prompts import SOLVE_PROMPT
from systems.common import finish, solve_once

THIRD_MODEL = "qwen2.5:7b"
TAGS = ("seeded_error", "consistent_correct", "always_wrong")


def default_models():
  """Solver, auditor, then one further open model. Duplicates are dropped."""
  seen = []
  for model in (SOLVER_MODEL, AUDITOR_MODEL, THIRD_MODEL):
    if model not in seen:
      seen.append(model)
  return seen


def model_slug(model):
  """Filesystem-safe model name. Windows rejects ':' in a path."""
  slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model).strip("-")
  return slug or "model"


def fingerprint(model):
  """Changes when the prompt, temperature, or model changes."""
  blob = json.dumps([SOLVE_PROMPT, SOLVER_TEMP, model], sort_keys=True)
  return hashlib.sha256(blob.encode()).hexdigest()[:12]


def out_path(run_id, split, model):
  return f"{RESULTS_DIR}/{run_id}_{split}_baseline-{model_slug(model)}.jsonl"


def metrics_path(run_id, split, model):
  return f"{RESULTS_DIR}/{run_id}_{split}_baseline-{model_slug(model)}_metrics.json"


def run_baseline(item, run_id, model=None):
  """One solve call. The parsed answer is both the initial and the final answer."""
  model = model or SOLVER_MODEL
  tr = Tracker()
  _text, answer = solve_once(item, tr, run_id, model=model)
  return finish(item, "baseline", tr, answer, answer, {"model": model})


def load_jsonl(path):
  rows = []
  with open(path, encoding="utf-8") as f:
    for line in f:
      if line.strip():
        rows.append(json.loads(line))
  return rows


def stats(rows):
  n = len(rows)
  empty = {"n": 0, "accuracy": None, "unparsed_rate": None,
           "avg_calls": None, "avg_tokens": None, "avg_latency_s": None}
  if n == 0:
    return empty
  return {
      "n": n,
      "accuracy": round(sum(bool(r["final_correct"]) for r in rows) / n, 4),
      "unparsed_rate": round(sum(bool(r["initial_missing"]) for r in rows) / n, 4),
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
      "config": {"temperature": SOLVER_TEMP, "prompt": "SOLVE_PROMPT"},
  }


def _fmt_pct(value):
  return "n/a" if value is None else f"{value:.1%}"


def _fmt_num(value):
  return "n/a" if value is None else f"{value:.1f}"


def print_report(metrics, run_id, split):
  ov = metrics["overall"]
  print(f"\n{'=' * 64}")
  print(f"  Baseline - {run_id} / {split} / {metrics['model']}")
  print(f"  temperature={metrics['config']['temperature']}  prompt={metrics['config']['prompt']}")
  print(f"{'=' * 64}")
  print(f"  Overall    n={ov['n']:4d}  acc={_fmt_pct(ov['accuracy'])}  "
        f"unparsed={_fmt_pct(ov['unparsed_rate'])}  "
        f"avg_calls={_fmt_num(ov['avg_calls'])}  "
        f"avg_tokens={_fmt_num(ov['avg_tokens'])}  "
        f"avg_latency_s={_fmt_num(ov['avg_latency_s'])}")
  print(f"{'-' * 64}")
  for tag, row in metrics["by_tag"].items():
    if row["n"] == 0:
      continue
    print(f"  {tag:<22} n={row['n']:4d}  acc={_fmt_pct(row['accuracy'])}  "
          f"unparsed={_fmt_pct(row['unparsed_rate'])}")
  print(f"{'=' * 64}\n")


def print_comparison(all_metrics):
  if len(all_metrics) < 2:
    return
  print(f"\n{'=' * 72}")
  print("  Baseline comparison")
  print(f"  {'model':<22}{'n':>6}{'acc':>10}{'unparsed':>12}{'tokens':>10}{'seconds':>10}")
  for metrics in all_metrics:
    ov = metrics["overall"]
    print(f"  {metrics['model']:<22}{ov['n']:6d}{_fmt_pct(ov['accuracy']):>10}"
          f"{_fmt_pct(ov['unparsed_rate']):>12}{_fmt_num(ov['avg_tokens']):>10}"
          f"{_fmt_num(ov['avg_latency_s']):>10}")
  print(f"{'=' * 72}\n")


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
      futs = {ex.submit(run_baseline, it, run_id, model): it for it in todo}
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
        mark = "missing" if result.initial_missing else ("right" if result.final_correct else "WRONG")
        print(f"  {finished}/{len(todo)} {it['id']:<12} {mark:<7} answer={result.final_answer}")
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


def parse_models(text):
  models = []
  for part in text.split(","):
    name = part.strip()
    if name and name not in models:
      models.append(name)
  if not models:
    sys.exit("Pass at least one model to --models.")
  return models


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--split", choices=["dev", "test"], default="dev")
  ap.add_argument("--run-id", default=None,
                  help="cache and result namespace; defaults to baseline_<split>")
  ap.add_argument("--models", default=",".join(default_models()),
                  help="comma-separated Ollama models (default: solver, auditor, qwen2.5:7b)")
  ap.add_argument("--limit", type=int, default=None,
                  help="evaluate a prefix of the shuffled split")
  ap.add_argument("--workers", type=int, default=WORKERS)
  ap.add_argument("--final", action="store_true", help="required for the test split")
  ap.add_argument("--report", action="store_true",
                  help="print saved results for --models, no model calls")
  args = ap.parse_args()

  run_id = args.run_id or f"baseline_{args.split}"
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
