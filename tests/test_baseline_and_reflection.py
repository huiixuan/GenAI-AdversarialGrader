"""Simulated baseline and self-reflection runs. No Ollama calls.

A stand-in model returns scripted answers so the full eval loop can be checked:
grading, cost, resume, fingerprint refusal, and the shared first-answer cache key.
"""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import SOLVER_MODEL, SOLVER_TEMP
from prompts import SELF_CRITIQUE_PROMPT, SELF_REVISE_PROMPT, SOLVE_PROMPT
import systems.adaptive_auditor as adaptive
import systems.adversarial as adversarial
import systems.baseline as baseline
import systems.self_reflection as reflection


def cache_key(model, messages, temperature, run_id, tag, system=None):
  """Same key call_llm writes. Kept here so the test fails if that key changes."""
  return hashlib.sha256(json.dumps(
      [model, system, messages, temperature, run_id, tag],
      sort_keys=True).encode()).hexdigest()


class SimulatedModel:
  """Scripted solver. The question text selects the script; the tag selects the stage."""

  def __init__(self):
    self.calls = []

  def __call__(self, tracker, *, model, messages, system=None, temperature=SOLVER_TEMP,
               max_tokens=1024, run_id="default", tag=""):
    self.calls.append({
        "model": model, "system": system, "messages": messages,
        "temperature": temperature, "run_id": run_id, "tag": tag,
    })
    tracker.calls += 1
    tracker.in_tokens += 8
    tracker.out_tokens += 4
    tracker.latency_s += 0.05
    tracker.log.append({"tag": tag, "model": model})
    question = messages[0]["content"]
    if tag == "solve" or tag.startswith("solve_"):
      if "FAILCALL" in question:
        raise RuntimeError("model down")
      if "UNPARSED" in question:
        return "I do not know"
      if "HALF" in question:
        return "ANSWER: 1/2"
      if "WRONG" in question:
        return "ANSWER: 4"
      return "ANSWER: 10"
    if tag.startswith("audit"):
      return "not json"
    if tag == "self-critique":
      if "EMPTY" in question:
        return "   "
      return "The total is 10, so an answer of 4 is wrong."
    if tag == "self-revise":
      if "NOANSWER" in question:
        return "still unsure"
      return "ANSWER: 10"
    raise AssertionError(f"unexpected tag {tag}")


def sample_items():
  return [
      {"id": "ok", "question": "OK What is 6+4?", "gold": "10",
       "tag": "consistent_correct", "source": "GSM8K", "level": None},
      {"id": "wrong", "question": "WRONG What is 6+4?", "gold": "10",
       "tag": "seeded_error", "source": "GSM8K", "level": None},
      {"id": "half", "question": "HALF of one", "gold": "0.5",
       "tag": "consistent_correct", "source": "GSM8K", "level": None},
      {"id": "blank", "question": "UNPARSED question", "gold": "3",
       "tag": "always_wrong", "source": "MATH", "level": 2},
      {"id": "empty", "question": "EMPTY WRONG critique", "gold": "10",
       "tag": "seeded_error", "source": "GSM8K", "level": None},
      {"id": "noans", "question": "NOANSWER WRONG revise", "gold": "10",
       "tag": "seeded_error", "source": "MATH", "level": 3},
      {"id": "down", "question": "FAILCALL please", "gold": "1",
       "tag": "always_wrong", "source": "MATH", "level": 4},
  ]


class SimulatedEvalTest(unittest.TestCase):
  def setUp(self):
    self.model = SimulatedModel()
    self.tmp = tempfile.TemporaryDirectory()
    self.data = os.path.join(self.tmp.name, "data")
    self.results = os.path.join(self.tmp.name, "results")
    os.makedirs(self.data)
    with open(os.path.join(self.data, "dev.json"), "w", encoding="utf-8") as handle:
      json.dump(sample_items(), handle)
    self.patches = [
        patch("systems.common.call_llm", self.model),
        patch("systems.self_reflection.call_llm", self.model),
        patch("systems.auditor.call_llm", self.model),
        patch("systems.adaptive_auditor.call_llm", self.model),
        patch("systems.baseline.check_ready"),
        patch("systems.self_reflection.check_ready"),
        patch("systems.baseline.DATA_DIR", self.data),
        patch("systems.self_reflection.DATA_DIR", self.data),
        patch("systems.baseline.RESULTS_DIR", self.results),
        patch("systems.self_reflection.RESULTS_DIR", self.results),
    ]
    for item in self.patches:
      item.start()
    self.addCleanup(self.tmp.cleanup)
    for item in self.patches:
      self.addCleanup(item.stop)
    self._argv = sys.argv

  def tearDown(self):
    sys.argv = self._argv

  def test_baseline_is_one_call_and_keeps_that_answer(self):
    item = sample_items()[1]
    result = baseline.run_baseline(item, "sim", model="qwen2.5:7b")
    self.assertEqual(result.system, "baseline")
    self.assertEqual([c["tag"] for c in self.model.calls], ["solve"])
    self.assertEqual(self.model.calls[0]["model"], "qwen2.5:7b")
    self.assertEqual(self.model.calls[0]["temperature"], SOLVER_TEMP)
    self.assertEqual(self.model.calls[0]["messages"][0]["content"],
                     SOLVE_PROMPT.format(q=item["question"]))
    self.assertEqual(result.initial_answer, "4")
    self.assertEqual(result.final_answer, "4")
    self.assertFalse(result.initial_correct)
    self.assertFalse(result.final_correct)
    self.assertEqual(result.calls, 1)
    self.assertEqual(result.extra, {"model": "qwen2.5:7b"})

  def test_default_baseline_model_is_the_configured_solver(self):
    result = baseline.run_baseline(sample_items()[0], "sim")
    self.assertEqual(result.final_answer, "10")
    self.assertTrue(result.final_correct)
    self.assertEqual(result.extra["model"], SOLVER_MODEL)

  def test_self_reflection_critiques_then_revises_with_the_same_model(self):
    item = sample_items()[1]
    result = reflection.run_self_reflection(item, "sim", model="llama3.1:8b")
    tags = [c["tag"] for c in self.model.calls]
    self.assertEqual(tags, ["solve", "self-critique", "self-revise"])
    self.assertTrue(all(c["model"] == "llama3.1:8b" for c in self.model.calls))
    self.assertTrue(all(c["temperature"] == SOLVER_TEMP for c in self.model.calls))
    solve, critique, revise = self.model.calls
    self.assertEqual(critique["messages"][-1], {"role": "user", "content": SELF_CRITIQUE_PROMPT})
    self.assertEqual(revise["messages"][-2]["content"],
                     "The total is 10, so an answer of 4 is wrong.")
    self.assertEqual(revise["messages"][-1], {"role": "user", "content": SELF_REVISE_PROMPT})
    self.assertEqual(result.initial_answer, "4")
    self.assertEqual(result.final_answer, "10")
    self.assertFalse(result.initial_correct)
    self.assertTrue(result.final_correct)
    self.assertTrue(result.extra["revised"])
    self.assertTrue(result.extra["changed"])
    self.assertEqual(result.calls, 3)

  def test_empty_critique_skips_revision_and_unparsed_revision_keeps_the_first_answer(self):
    empty = reflection.run_self_reflection(sample_items()[4], "sim")
    self.assertEqual([c["tag"] for c in self.model.calls], ["solve", "self-critique"])
    self.assertEqual(empty.final_answer, "4")
    self.assertFalse(empty.extra["revised"])
    self.assertEqual(empty.extra["revision_failure"], "critique was empty")

    self.model.calls.clear()
    blocked = reflection.run_self_reflection(sample_items()[5], "sim")
    self.assertEqual([c["tag"] for c in self.model.calls],
                     ["solve", "self-critique", "self-revise"])
    self.assertEqual(blocked.initial_answer, blocked.final_answer)
    self.assertEqual(blocked.final_answer, "4")
    self.assertTrue(blocked.extra["revision_blocked"])
    self.assertFalse(blocked.extra["changed"])

  def test_first_answer_cache_key_matches_across_the_two_systems(self):
    item = sample_items()[0]
    baseline.run_baseline(item, "shared", model=SOLVER_MODEL)
    reflection.run_self_reflection(item, "shared", model=SOLVER_MODEL)
    solve_calls = [c for c in self.model.calls if c["tag"] == "solve"]
    self.assertEqual(solve_calls[0]["messages"], solve_calls[1]["messages"])
    keys = [cache_key(c["model"], c["messages"], c["temperature"], c["run_id"], c["tag"], c["system"])
            for c in solve_calls]
    self.assertEqual(keys[0], keys[1])
    critique = next(c for c in self.model.calls if c["tag"] == "self-critique")
    critique_key = cache_key(critique["model"], critique["messages"], critique["temperature"],
                             critique["run_id"], critique["tag"], critique["system"])
    self.assertNotEqual(critique_key, keys[0])

  def test_all_four_systems_share_one_solve_cache_key(self):
    item = sample_items()[0]
    run_id = "shared_dev"
    self.model.calls.clear()
    baseline.run_baseline(item, run_id)
    reflection.run_self_reflection(item, run_id)
    adversarial.run_adversarial(item, run_id)
    adaptive.run_one(item, run_id)
    solve_calls = [c for c in self.model.calls if c["tag"] == "solve"]
    self.assertEqual(len(solve_calls), 4)
    keys = [cache_key(c["model"], c["messages"], c["temperature"], c["run_id"], c["tag"], c["system"])
            for c in solve_calls]
    self.assertEqual(len(set(keys)), 1)
    self.assertTrue(all(c["model"] == SOLVER_MODEL and c["run_id"] == run_id for c in solve_calls))
    extra = [c for c in self.model.calls if c["tag"].startswith("solve_")]
    self.assertEqual(len(extra), 2)
    extra_keys = [cache_key(c["model"], c["messages"], c["temperature"], c["run_id"], c["tag"], c["system"])
                  for c in extra]
    self.assertTrue(all(k != keys[0] for k in extra_keys))

  def test_baseline_eval_grades_resumes_and_refuses_a_changed_setup(self):
    self._run("baseline", ["--split", "dev", "--run-id", "sim", "--models", "llama3.2:3b,qwen2.5:7b"])
    rows = self._rows("sim_dev_baseline-llama3.2-3b.jsonl")
    self.assertEqual({r["qid"] for r in rows},
                     {"ok", "wrong", "half", "blank", "empty", "noans"})
    by_id = {r["qid"]: r for r in rows}
    self.assertTrue(by_id["ok"]["final_correct"])
    self.assertTrue(by_id["half"]["final_correct"])
    self.assertEqual(by_id["half"]["final_answer"], "1/2")
    self.assertFalse(by_id["wrong"]["final_correct"])
    self.assertTrue(by_id["blank"]["initial_missing"])
    self.assertEqual(by_id["ok"]["initial_answer"], by_id["ok"]["final_answer"])
    self.assertTrue(all(r["system"] == "baseline" and r["calls"] == 1 for r in rows))
    self.assertTrue(all(":" not in os.path.basename(name) for name in os.listdir(self.results)))

    metrics = self._metrics("sim_dev_baseline-llama3.2-3b_metrics.json")
    self.assertEqual(metrics["overall"]["n"], 6)
    self.assertEqual(metrics["overall"]["accuracy"], round(2 / 6, 4))
    self.assertEqual(metrics["overall"]["unparsed_rate"], round(1 / 6, 4))
    self.assertEqual(metrics["by_tag"]["consistent_correct"]["n"], 2)
    self.assertEqual(metrics["by_tag"]["consistent_correct"]["accuracy"], 1.0)
    self.assertEqual(metrics["by_tag"]["seeded_error"]["accuracy"], 0.0)
    self.assertEqual(metrics["by_tag"]["always_wrong"]["n"], 1)

    done = len(self.model.calls)
    self._run("baseline", ["--split", "dev", "--run-id", "sim", "--models", "llama3.2:3b"])
    # The failed question is retried. The six finished questions are not.
    self.assertEqual(len(self.model.calls) - done, 1)
    self.assertEqual(self.model.calls[-1]["tag"], "solve")
    self.assertEqual(len(self._rows("sim_dev_baseline-llama3.2-3b.jsonl")), 6)

    path = os.path.join(self.results, "sim_dev_baseline-llama3.2-3b.jsonl")
    saved = baseline.load_jsonl(path)
    saved[0]["extra"]["fingerprint"] = "changed"
    with open(path, "w", encoding="utf-8") as handle:
      for row in saved:
        handle.write(json.dumps(row) + "\n")
    with self.assertRaises(SystemExit) as raised:
      self._run("baseline", ["--split", "dev", "--run-id", "sim", "--models", "llama3.2:3b"])
    self.assertIn("different setup", str(raised.exception))

  def test_self_reflection_eval_reports_fixes_and_regressions(self):
    self._run("reflection", ["--split", "dev", "--run-id", "sim", "--models", "llama3.2:3b"])
    rows = {r["qid"]: r for r in self._rows("sim_dev_self-reflection-llama3.2-3b.jsonl")}
    self.assertEqual(set(rows), {"ok", "wrong", "half", "blank", "empty", "noans"})
    self.assertEqual(rows["wrong"]["initial_answer"], "4")
    self.assertEqual(rows["wrong"]["final_answer"], "10")
    self.assertTrue(rows["wrong"]["extra"]["changed"])
    self.assertEqual(rows["empty"]["final_answer"], "4")
    self.assertFalse(rows["empty"]["extra"]["revised"])
    self.assertEqual(rows["noans"]["final_answer"], "4")
    self.assertTrue(rows["noans"]["extra"]["revision_blocked"])
    self.assertEqual(rows["half"]["initial_answer"], "1/2")
    self.assertEqual(rows["half"]["final_answer"], "10")
    self.assertTrue(rows["ok"]["initial_correct"] and rows["ok"]["final_correct"])

    metrics = self._metrics("sim_dev_self-reflection-llama3.2-3b_metrics.json")
    overall = metrics["overall"]
    self.assertEqual(overall["n"], 6)
    self.assertEqual(overall["initial_accuracy"], round(2 / 6, 4))
    self.assertEqual(overall["final_accuracy"], round(2 / 6, 4))
    self.assertEqual(overall["fixed"], 1)
    self.assertEqual(overall["broke"], 1)
    self.assertEqual(overall["changed_rate"], 0.5)
    self.assertEqual(overall["avg_calls"], round(17 / 6, 2))
    self.assertEqual(metrics["by_tag"]["seeded_error"]["fixed"], 1)
    self.assertEqual(metrics["by_tag"]["consistent_correct"]["broke"], 1)

    done = len(self.model.calls)
    self._run("reflection", ["--report", "--split", "dev", "--run-id", "sim", "--models", "llama3.2:3b"])
    self.assertEqual(len(self.model.calls), done)

  def test_test_split_is_sealed_for_both_systems(self):
    for module in ("baseline", "reflection"):
      with self.assertRaises(SystemExit) as raised:
        self._run(module, ["--split", "test", "--run-id", "sealed", "--models", "llama3.2:3b"])
      self.assertIn("sealed", str(raised.exception))
    self.assertEqual(self.model.calls, [])

  def test_report_without_results_stops(self):
    with self.assertRaises(SystemExit) as raised:
      self._run("baseline", ["--report", "--split", "dev", "--run-id", "missing", "--models", "llama3.2:3b"])
    self.assertIn("no result files", str(raised.exception))

  def _run(self, module, args):
    sys.argv = [module, *args]
    {"baseline": baseline.main, "reflection": reflection.main}[module]()

  def _rows(self, name):
    return baseline.load_jsonl(os.path.join(self.results, name))

  def _metrics(self, name):
    with open(os.path.join(self.results, name), encoding="utf-8") as handle:
      return json.load(handle)


if __name__ == "__main__":
  unittest.main()
