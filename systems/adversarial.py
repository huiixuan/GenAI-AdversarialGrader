"""System 3: solver answers, a separate auditor checks it, then may revise.

Run the evaluation with:
    python -m systems.adversarial --run-id final-auditor --split test --final
"""
from config import SOLVER_MODEL, SOLVER_TEMP
from grading import extract_model_answer
from llm import Tracker, call_llm
from prompts import REVISE_PROMPT, SOLVE_PROMPT
from systems.auditor import AuditorConfig, critique_solver, run_audit
from systems.common import finish, solve_once


def run_adversarial(item, run_id, cfg=None):
  cfg = cfg or AuditorConfig()
  tr = Tracker()
  first_response, first = solve_once(item, tr, run_id)
  extra = {"variant": cfg.tag(), "auditor_model": cfg.model}
  audit = run_audit(item["question"], first, tr, run_id, cfg)
  verdict = audit["verdict"]
  extra.update(steps=audit["steps"], tool_calls=audit["tool_calls"],
               failure=audit["failure"], safe_stop=audit.get("safe_stop", False))
  extra["verdict"] = verdict

  final, revised = first, False
  if verdict is not None:
    critique = critique_solver(
      item["question"], first_response, verdict, extra.get("tool_calls", []), tr, run_id, cfg)
    extra["critique"] = critique
    extra["critique_generated"] = bool(critique.strip())
    if extra["critique_generated"]:
      revision_prompt = REVISE_PROMPT.replace("{critique}", critique)
      extra["revision_attempted"] = True
      revision = call_llm(
        tr, model=SOLVER_MODEL,
        messages=[
          {"role": "user", "content": SOLVE_PROMPT.format(q=item["question"])},
          {"role": "assistant", "content": first_response},
          {"role": "user", "content": revision_prompt},
        ],
        temperature=SOLVER_TEMP, run_id=run_id, tag=f"revise-{cfg.tag()}")
      revised_answer = extract_model_answer(revision)
      extra["revision_attempted"] = True
      extra["revision_parsed"] = revised_answer is not None
      if revised_answer is not None:
        final, revised = revised_answer, True
      else:
        extra["revision_blocked"] = True
        extra["revision_failure"] = "solver retry had no parsable final answer"
    else:
      extra["revision_attempted"] = False
      extra["revision_blocked"] = True
      extra["revision_failure"] = "auditor returned an empty critique"
  else:
    extra["revision_attempted"] = False
    extra["revision_blocked"] = True
    extra["revision_failure"] = "auditor did not produce a usable verdict"
  extra.update(revised=revised, changed=final != first)
  return finish(item, "adversarial", tr, first, final, extra)


def main():
  """Use the shared evaluation runner for dataset, reporting, and test safeguards."""
  from scripts.eval_adversarial import main as eval_main

  eval_main()


if __name__ == "__main__":
  main()