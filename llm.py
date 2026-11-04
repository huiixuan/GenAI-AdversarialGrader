"""The ONLY module that talks to the model (a local Ollama server).

Every call is cached on disk and logged to a per-question Tracker (calls, tokens,
latency, dollars). The run_id and tag are part of the cache key:
  * pilot tagging uses run_id="pilot"
  * each evaluation run should use its own run_id ("eval1", "eval2", ...)
so the two never share cached answers.
"""
import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field

from config import (MAX_TOKENS, NUM_CTX, OLLAMA_HOST, OLLAMA_TIMEOUT, PRICES,
                    RESULTS_DIR, SOLVER_TEMP)

CACHE_PATH = f"{RESULTS_DIR}/llm_cache.jsonl"


@dataclass
class Tracker:
  """Accumulates cost for ONE question under ONE system."""
  calls: int = 0
  in_tokens: int = 0
  out_tokens: int = 0
  latency_s: float = 0.0        # sequential wall-clock within a question
  usd: float = 0.0              # only non-zero if config.PRICES has an (assumed) price
  log: list = field(default_factory=list)   # prompts/responses, for debugging


_cache = {}
_cache_loaded = False
_cache_lock = threading.Lock()
_client = None                    # tests replace this with a fake


def _get_client():
  global _client
  if _client is None:
    import ollama
    _client = ollama.Client(host=OLLAMA_HOST, timeout=OLLAMA_TIMEOUT)
  return _client


def _strip_thinking(text):
  """Reasoning models may emit <think>...</think>; the answer parser should not see it."""
  return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _generate(model, messages, system, temperature, max_tokens, seed):
  """One raw Ollama call. Returns (text, in_tokens, out_tokens)."""
  msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
  resp = _get_client().chat(model=model, messages=msgs, options={
      "temperature": temperature, "num_predict": max_tokens,
      "num_ctx": NUM_CTX, "seed": seed})
  text = _strip_thinking(resp["message"]["content"])
  return text, int(resp["prompt_eval_count"] or 0), int(resp["eval_count"] or 0)


def check_ready(model):
  """Fail early, with an actionable message, if Ollama isn't usable."""
  try:
    listing = _get_client().list()
  except Exception as e:
    raise SystemExit(f"Can't reach Ollama at {OLLAMA_HOST} ({e!r}).\n"
                      f"Start it with `ollama serve` (or open the Ollama app).")
  models = listing["models"] if not hasattr(listing, "models") else listing.models

  def name(m):
    if isinstance(m, dict):
      return m.get("model") or m.get("name")
    return getattr(m, "model", None) or getattr(m, "name", None)

  have = {name(m) for m in models}
  if model not in have and f"{model}:latest" not in have:
    raise SystemExit(f"Model {model!r} is not pulled. Run: ollama pull {model}\n"
                      f"Models you have: {sorted(n for n in have if n)}")


def _ensure_cache():
  global _cache_loaded
  with _cache_lock:
    if _cache_loaded:
      return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if os.path.exists(CACHE_PATH):
      for line in open(CACHE_PATH):
        try:
          r = json.loads(line)
          _cache[r["key"]] = r
        except Exception:
          pass
    _cache_loaded = True


def call_llm(tracker, *, model, messages, system=None, temperature=SOLVER_TEMP,
             max_tokens=MAX_TOKENS, run_id="default", tag=""):
  """Make (or replay from cache) one model call and return the response text.

  A cache hit still adds the ORIGINAL tokens/latency to the tracker, so cost
  reporting is identical whether or not a call was replayed.
  """
  _ensure_cache()
  key = hashlib.sha256(json.dumps(
      [model, system, messages, temperature, run_id, tag],
      sort_keys=True).encode()).hexdigest()
  # Distinct, reproducible sampling seed per (prompt, run_id, tag). Without this, the k pilot
  # samples of one question could be identical copies, and reruns wouldn't be reproducible.
  seed = int(key[:8], 16)

  rec = _cache.get(key)
  if rec is None:
    t0 = time.time()
    for attempt in range(3):
      try:
        text, n_in, n_out = _generate(model, messages, system,
                                      temperature, max_tokens, seed)
        break
      except Exception:
        if attempt == 2:
          raise
        time.sleep(2 ** attempt)
    rec = {"key": key, "text": text, "in_tokens": n_in, "out_tokens": n_out,
            "latency_s": time.time() - t0}
    with _cache_lock:
      with open(CACHE_PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")
      _cache[key] = rec

  p_in, p_out = PRICES.get(model, (0.0, 0.0))
  tracker.calls += 1
  tracker.in_tokens += rec["in_tokens"]
  tracker.out_tokens += rec["out_tokens"]
  tracker.latency_s += rec["latency_s"]
  tracker.usd += (rec["in_tokens"] * p_in + rec["out_tokens"] * p_out) / 1e6
  tracker.log.append({"tag": tag, "model": model, "system": system,
                      "messages": messages, "response": rec["text"]})
  return rec["text"]