#!/usr/bin/env python3
"""
Stage 2: pilot tagging. Calls the local Ollama model set in config.SOLVER_MODEL
(free, but the Ollama server must be running).

Run from the project root, AFTER scripts.prepare_data:
    ollama serve & ollama pull llama3.1:8b
    python -m scripts.pilot_tagging --limit 50      # cheap trial: check accuracy first
    python -m scripts.pilot_tagging                 # full run on every question
    python -m scripts.pilot_tagging --summary       # just print the summary, no calls

After the full run, read data/review_always_wrong.json, list the ids with wrong
gold labels or ambiguous wording in a text file (one id per line), then:
    python -m scripts.pilot_tag --exclude data/exclude_ids.txt

Reruns are cheap: every call is cached in results/llm_cache.jsonl.
"""
import argparse
import os

from config import DATA_DIR, PILOT_K, PILOT_TEMPERATURE, SOLVER_MODEL, WORKERS
from data_prep.tagging import apply_exclusions, run_pilot, summary
from llm import check_ready


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--limit", type=int, default=None,
                  help="tag only a random subset of N questions")
  ap.add_argument("--k", type=int, default=PILOT_K, help="samples per question")
  ap.add_argument("--model", default=SOLVER_MODEL)
  ap.add_argument("--temperature", type=float, default=PILOT_TEMPERATURE)
  ap.add_argument("--workers", type=int, default=WORKERS)
  ap.add_argument("--summary", action="store_true", help="print summary only, no API calls")
  ap.add_argument("--exclude", metavar="FILE",
                  help="remove the question ids listed in FILE (one per line) and exit")
  a = ap.parse_args()

  if not os.path.exists(f"{DATA_DIR}/all.json"):
    raise SystemExit("data/all.json not found. Run: python -m scripts.prepare_data")

  if a.exclude:
    ids = [l.strip() for l in open(a.exclude) if l.strip() and not l.startswith("#")]
    apply_exclusions(ids)
    return

  if a.summary:
    summary()
    return

  check_ready(a.model)

  items = run_pilot(a.limit, a.k, a.model, a.temperature, a.workers)
  summary(items)
  print(f"\nRead {DATA_DIR}/review_always_wrong.json and drop bad questions with --exclude.")


if __name__ == "__main__":
  main()