"""Sampling, stratified dev/test splitting, and reading/writing the dataset files."""
import json
import os
from collections import defaultdict

from config import (DATA_DIR, MATH_LEVELS, N_GSM8K, N_MATH, DEV_FRACTION)


def sample_items(gsm, math, rng, n_gsm8k=N_GSM8K, n_math=N_MATH):
  """Random n_gsm8k GSM8K questions + n_math MATH questions spread evenly over levels."""
  rng.shuffle(gsm)
  out = gsm[:n_gsm8k]
  if len(gsm) < n_gsm8k:
    print(f"[warn] GSM8K: only {len(gsm)} available (wanted {n_gsm8k})")
  per_level = n_math // len(MATH_LEVELS)
  for lv in MATH_LEVELS:
    pool = [m for m in math if m["level"] == lv]
    rng.shuffle(pool)
    if len(pool) < per_level:
      print(f"[warn] MATH level {lv}: only {len(pool)} available (wanted {per_level})")
    out += pool[:per_level]
  return out


def assign_splits(items, rng, dev_fraction=DEV_FRACTION):
  """Stratified dev/test split by (source, level). Mutates and returns items."""
  groups = defaultdict(list)
  for it in items:
    groups[(it["source"], it["level"])].append(it)
  for g in groups.values():
    rng.shuffle(g)
    n_dev = round(len(g) * dev_fraction)
    for k, it in enumerate(g):
      it["split"] = "dev" if k < n_dev else "test"
  return items


def _atomic_dump(obj, path):
  """Write to a temp file, then rename: a crash or Ctrl-C can never leave a half-written file."""
  tmp = path + ".tmp"
  with open(tmp, "w") as f:
    json.dump(obj, f, indent=2)
  os.replace(tmp, path)


def save_all(items):
  """Write all.json plus dev.json and test.json (atomically)."""
  os.makedirs(DATA_DIR, exist_ok=True)
  _atomic_dump(items, f"{DATA_DIR}/all.json")
  for sp in ("dev", "test"):
    _atomic_dump([x for x in items if x["split"] == sp], f"{DATA_DIR}/{sp}.json")


def load_all():
  return json.load(open(f"{DATA_DIR}/all.json"))