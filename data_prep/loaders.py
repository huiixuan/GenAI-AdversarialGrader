"""Download + filter GSM8K and MATH. Pure loading: no sampling or splitting."""
import re

from typing import Any, cast
from config import MATH_LEVELS
from grading import is_numeric_gold, last_boxed, normalize


def load_gsm8k(removed):
  from datasets import load_dataset
  ds = load_dataset("openai/gsm8k", "main", split="test")
  items = []
  for i, x in enumerate(ds):
    row = cast(dict[str, Any], x)
    gold = row["answer"].split("####")[-1].strip().replace(",", "")
    if not is_numeric_gold(gold):
      removed.append({"id": f"gsm_{i}", "reason": "non-numeric gold"})
      continue
    items.append({"id": f"gsm_{i}", "source": "GSM8K", "level": None,
                  "subject": "grade-school", "question": row["question"],
                  "gold": gold})
  return items


def _level_to_int(level):
  m = re.search(r"(\d)", str(level))
  return int(m.group(1)) if m else None


def load_math(removed):
  """Try the full MATH test set (5000); fall back to MATH-500."""
  from datasets import load_dataset, concatenate_datasets
  rows = []
  try:
    subjects = ["algebra", "counting_and_probability", "geometry",
                "intermediate_algebra", "number_theory", "prealgebra",
                "precalculus"]
    parts = [load_dataset("EleutherAI/hendrycks_math", s, split="test")
            for s in subjects]
    ds = concatenate_datasets(parts)
    for i, x in enumerate(ds):
      row = cast(dict[str, Any], x)
      rows.append({
        "idx": i, 
        "problem": row["problem"],
        "level": _level_to_int(row["level"]),
        "subject": row["type"],
        "gold": last_boxed(row["solution"])
      })
    print(f"[MATH] loaded full test set: {len(rows)} problems")
  except Exception as e:
    print(f"[MATH] full set failed ({e!r}); falling back to MATH-500")
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    for i, x in enumerate(ds):
      row = cast(dict[str, Any], x)
      rows.append({
        "idx": i, 
        "problem": row["problem"],
        "level": _level_to_int(row["level"]),
        "subject": row["subject"], 
        "gold": row["answer"]
      })

  items = []
  for r in rows:
    rid = f"math_{r['idx']}"
    if r["gold"] is None:
      removed.append({"id": rid, "reason": "no boxed gold"}); continue
    if "[asy]" in r["problem"]:
      removed.append({"id": rid, "reason": "diagram ([asy])"}); continue
    if r["level"] not in MATH_LEVELS:
      continue  # out-of-scope level: not a "removal", just not in design
    if not is_numeric_gold(r["gold"]):
      removed.append({"id": rid, "reason": "non-numeric gold"}); continue
    items.append({"id": rid, "source": "MATH", "level": r["level"],
                  "subject": r["subject"], "question": r["problem"],
                  "gold": normalize(r["gold"])})
  return items


def dedupe(items, removed):
  seen, out = set(), []
  for it in items:
    key = re.sub(r"\s+", " ", it["question"].strip().lower())
    if key in seen:
      removed.append({"id": it["id"], "reason": "duplicate question"})
      continue
    seen.add(key)
    out.append(it)
  return out