"""
Stage 1: build the dataset. Makes NO LLM calls (only downloads from Hugging Face).

Run from the project root:
    pip install datasets
    python -m scripts.prepare_data
    python -m scripts.prepare_data --n-gsm8k 50 --n-math 100     # smaller trial

Writes (in ./data):
    all.json, dev.json, test.json   the sampled questions with a fixed split
    removed_log.json                every filtered-out question and why

Next stage: python -m scripts.pilot_tag --limit 50
"""
import argparse
import json
import os
import random
from collections import Counter

from config import DATA_DIR, N_GSM8K, N_MATH, DEV_FRACTION, SEED
from data_prep.loaders import load_gsm8k, load_math, dedupe
from data_prep.splits import sample_items, assign_splits, save_all


def build(n_gsm8k, n_math, dev_fraction, seed):
  rng = random.Random(seed)
  removed = []

  gsm = dedupe(load_gsm8k(removed), removed)
  math = dedupe(load_math(removed), removed)
  print(f"after filtering: {len(gsm)} GSM8K, {len(math)} MATH candidates")

  items = sample_items(gsm, math, rng, n_gsm8k, n_math)
  items = assign_splits(items, rng, dev_fraction)
  return items, removed


def sanity_check(items):
  """Cheap checks that catch silent mistakes before you spend money on API calls."""
  ids = [x["id"] for x in items]
  assert len(ids) == len(set(ids)), "duplicate ids in sample"
  assert all(x["gold"] != "" for x in items), "empty gold answer"
  assert {x["split"] for x in items} <= {"dev", "test"}, "bad split label"
  assert not any("[asy]" in x["question"] for x in items), "diagram problem slipped through"


def report(items, removed):
  print(f"\nsaved {len(items)} questions to {DATA_DIR}/")
  print("  by split:        ", dict(Counter(x["split"] for x in items)))
  print("  by source/level: ", dict(sorted(Counter(
    (x["source"], x["level"]) for x in items).items(), key=str)))
  print("  by dev/test x source:", dict(Counter((x["split"], x["source"]) for x in items)))
  print(f"\nremoved {len(removed)} questions during filtering:")
  
  for reason, n in Counter(r["reason"] for r in removed).most_common():
    print(f"  {n:>5}  {reason}")
  print("\nthree random examples to eyeball:")
  
  for x in random.Random(0).sample(items, min(3, len(items))):
    print(f"  [{x['id']}] gold={x['gold']!r}  {x['question'][:110].strip()!r}...")


def main():
  ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--n-gsm8k", type=int, default=N_GSM8K)
  ap.add_argument("--n-math", type=int, default=N_MATH)
  ap.add_argument("--dev-fraction", type=float, default=DEV_FRACTION)
  ap.add_argument("--seed", type=int, default=SEED)
  ap.add_argument("--force", action="store_true",
                  help="overwrite an existing dataset (WARNING: discards pilot tags)")
  a = ap.parse_args()

  if os.path.exists(f"{DATA_DIR}/all.json") and not a.force:
    raise SystemExit(f"{DATA_DIR}/all.json already exists. Rebuilding would change the "
                      f"split and discard pilot tags. Use --force if you really mean it.")

  items, removed = build(a.n_gsm8k, a.n_math, a.dev_fraction, a.seed)
  sanity_check(items)
  save_all(items)
  json.dump(removed, open(f"{DATA_DIR}/removed_log.json", "w"), indent=2)
  report(items, removed)
  print("\nNext: python -m scripts.pilot_tag --limit 50")


if __name__ == "__main__":
  main()