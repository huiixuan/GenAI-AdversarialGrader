"""Pieces every system shares: the Result record and the first solver call."""
from dataclasses import dataclass, field

from config import SOLVER_MODEL, SOLVER_TEMP
from grading import extract_model_answer, is_correct
from llm import call_llm
from prompts import SOLVE_PROMPT


@dataclass
class Result:
  qid: str
  system: str
  initial_answer: str | None     # the shared first answer, before any critique or revision
  final_answer: str | None
  initial_correct: bool
  final_correct: bool
  initial_missing: bool          # the first response had no parsable answer at all
  calls: int
  in_tokens: int
  out_tokens: int
  latency_s: float
  usd: float
  extra: dict = field(default_factory=dict)   # audit verdict, tool calls, flags...
  log: list = field(default_factory=list)     # every prompt/response, for debugging


def solve_once(item, tracker, run_id):
  """The solver's first answer. Same model, prompt, run_id and tag in every system, so within one
  run_id all systems start from the SAME cached first answer for a given question.
  Returns (response_text, parsed_answer)."""
  messages = [{"role": "user", "content": SOLVE_PROMPT.format(q=item["question"])}]
  text = call_llm(tracker, model=SOLVER_MODEL, messages=messages, temperature=SOLVER_TEMP,
                  run_id=run_id, tag="solve")
  return text, extract_model_answer(text)


def finish(item, system, tracker, initial, final, extra=None):
  return Result(
    qid=item["id"], 
    system=system, 
    initial_answer=initial, 
    final_answer=final,
    initial_correct=is_correct(initial, item["gold"]),
    final_correct=is_correct(final, item["gold"]),
    initial_missing=initial is None,
    calls=tracker.calls, 
    in_tokens=tracker.in_tokens, 
    out_tokens=tracker.out_tokens,
    latency_s=tracker.latency_s, 
    usd=tracker.usd, 
    extra=extra or {}, 
    log=tracker.log
  )