"""Independently solve a math problem, then critique the solver's work.

The auditor's first solve is blind. After it returns an answer, the solver's original response is
shared in a separate critique call; code compares the answers and labels the result.
"""
from dataclasses import dataclass
from fractions import Fraction
import json
import re

from config import AUDITOR_MAX_STEPS, AUDITOR_MAX_TOKENS, AUDITOR_MODEL, AUDITOR_TEMP
from grading import is_correct, to_fraction
from llm import call_llm
from prompts import (AUDITOR_CRITIQUE_PROMPT, AUDITOR_NEXT, AUDITOR_NEXT_LAST,
                     AUDITOR_TOOL_RESULT, build_auditor_system, build_auditor_user)
from tools import available_modules, extract_marked_answer, run_tool

AUDITOR_PIPELINE_VERSION = "2026-11-09-blind-solve-critique-v5"


@dataclass
class AuditorConfig:
  """One audit setup. Ablations are just different configs."""
  model: str = AUDITOR_MODEL
  temperature: float = AUDITOR_TEMP
  max_steps: int = AUDITOR_MAX_STEPS
  max_tokens: int = AUDITOR_MAX_TOKENS
  use_tools: bool = True          # False: the "no tools" ablation
  style: str = "adversarial"      # "neutral": the "neutral prompt" ablation

  def tag(self):
    return (f"{self.style}-t{int(self.use_tools)}"
            f"-n{self.max_steps}-blind-critique")


_APPROXIMATE_EVIDENCE = re.compile(
  r"\b(?:linspace|unique|monte\s*carlo|sampling|sampled|round(?:ed|ing)?|"
  r"floating[- ]point|numerical\s+approx|random)\b", re.IGNORECASE)


def _extract_json(text):
  """Parse a JSON object from the auditor reply, including fenced or prefaced JSON."""
  if not text:
    return None
  content = re.sub(r"```(?:json)?", "", text)
  decoder = json.JSONDecoder()
  for match in re.finditer(r"\{", content):
    try:
      result, _ = decoder.raw_decode(content[match.start():])
      return result
    except ValueError:
      continue
  return None


def parse_verdict(obj):
  """Normalise the auditor's independent solution."""
  claims = []
  for c in obj.get("claims") or []:
    if not isinstance(c, dict):
      continue
    status = str(c.get("status", "")).lower()
    if status not in ("pass", "fail", "unsupported"):
      status = "unsupported"
    claims.append({
      "text": str(c.get("text", ""))[:300], 
      "status": status,
      "evidence": str(c.get("evidence", ""))[:500]
    })
  flawed = str(obj.get("overall", "")).lower() == "flawed"
  overall = "flawed" if flawed else "ok"
  return {"claims": claims, "overall": overall, "model_overall": overall,
          "auditor_answer": str(obj.get("auditor_answer", "")).strip()[:100],
          "candidate_answer": str(obj.get("auditor_answer", "")).strip()[:100],
          "target": str(obj.get("target", "")).strip()[:150],
          "problem_type": str(obj.get("problem_type", "")).strip()[:80],
          "interpretation": str(obj.get("interpretation", "")).strip()[:500],
          "independence": str(obj.get("independence", "")).strip()[:500],
          "derivation": str(obj.get("derivation", "")).strip()[:1000],
          "verification_plan": str(obj.get("verification_plan", "")).strip()[:500],
          "confidence": str(obj.get("confidence", "")).strip().lower()[:20],
          "evidence_backed": all(
              c["status"] != "unsupported" and
              (c["status"] != "pass" or bool(c["evidence"]))
              for c in claims),
          "approximate_evidence": False,
          "decision": "model", "grounded": None}


def _matching_checks(answer, tool_calls):
  """Return successful, non-echo tool checks whose marked result matches answer."""
  target = to_fraction(answer) if answer else None
  if target is None:
    return []
  tol = Fraction(1, 10**9) * max(1, abs(target))
  checks = []
  seen = set()
  for tc in tool_calls:
    output = tc["output"]
    if output.startswith("ERROR") or tc.get("echo"):
      continue
    call_key = (tc.get("tool"), tc.get("input"))
    if call_key in seen:
      continue
    marked = extract_marked_answer(output)
    value = to_fraction(marked) if marked is not None else None
    if value is None or abs(value - target) > tol:
      continue
    seen.add(call_key)
    checks.append(tc)
  return checks


def _has_method_diversity(checks):
  """Require both execution backends for a grounded disagreement."""
  return {tc.get("tool") for tc in checks} >= {"calculator", "python"}


def _has_approximate_evidence(verdict, tool_calls):
  text = " ".join([
      verdict.get("interpretation", ""),
      verdict.get("independence", ""),
      verdict.get("derivation", ""),
      *(c.get("evidence", "") for c in verdict.get("claims", [])),
      *(tc.get("input", "") for tc in tool_calls),
  ])
  return bool(_APPROXIMATE_EVIDENCE.search(text))


def _tool_evidence_consistent(answer, tool_calls):
  """If tools were used, require every call to succeed and support the reported answer."""
  if not tool_calls:
    return True
  expected = to_fraction(answer) if answer else None
  if expected is None:
    return False
  tolerance = Fraction(1, 10**9) * max(1, abs(expected))
  for call in tool_calls:
    output = call["output"]
    if output.startswith("ERROR") or output.startswith("(no output"):
      return False
    marked = extract_marked_answer(output)
    actual = to_fraction(marked) if marked is not None else None
    if actual is None or abs(actual - expected) > tolerance:
      return False
  return True


def _tool_error(action, input_text, seen_calls):
  """Reject predictable bad requests before spending a sandbox/tool call."""
  text = str(input_text).strip()
  if not text:
    return ("ERROR: empty input. Provide a complete independent computation. "
            "For Python use real lines and end with "
            "print('AUDITOR_ANSWER: ' + str(result)).")
  if action == "calculator":
    names = set(re.findall(r"\b[A-Za-z_]\w*\b", text))
    allowed = {"sqrt", "abs", "round", "floor", "ceil", "factorial", "comb", "perm",
                "gcd", "lcm", "log", "log10", "log2", "exp", "sin", "cos", "tan",
                "min", "max", "pi", "e"}
    if names - allowed:
      return ("ERROR: calculator accepts only a numeric expression. "
              "For variables, equations, symbolic algebra, polynomial remainders, "
              "or sequences, use Python with a complete program and a quoted "
              "AUDITOR_ANSWER marker.")
    if "=" in text and "==" not in text:
      return ("ERROR: calculator cannot solve equations. Use Python to assign variables "
              "and solve the equation.")
  key = (action, text)
  if key in seen_calls:
    return ("ERROR: repeated tool request. Change the method and input; do not repeat "
            "the failed request.")
  return None


def decide(verdict, solver_answer, tool_calls):
  """Compare the blind independent solution with the solver answer."""
  aud = verdict["auditor_answer"]
  checks = _matching_checks(aud, tool_calls) if verdict.get("target") else []
  verdict["grounding_checks"] = len(checks)
  verdict["method_diverse"] = _has_method_diversity(checks)
  verdict["approximate_evidence"] = _has_approximate_evidence(verdict, tool_calls)
  verdict["tool_evidence_consistent"] = _tool_evidence_consistent(aud, tool_calls)
  auditor_value = to_fraction(aud) if aud else None
  solver_value = to_fraction(solver_answer) if solver_answer else None
  comparable = auditor_value is not None and solver_value is not None
  differs = comparable and not is_correct(solver_answer, aud)
  claims_valid = verdict.get("evidence_backed", False)
  verdict["grounded"] = (bool(verdict.get("target"))
                          and bool(verdict.get("problem_type"))
                          and bool(verdict.get("interpretation"))
                          and bool(verdict.get("derivation"))
                          and verdict.get("model_overall") == "ok"
                          and verdict.get("confidence") == "high"
                          and claims_valid
                          and not verdict["approximate_evidence"]
                          and verdict["tool_evidence_consistent"]
                          and comparable)
  verdict["candidate_disagreement"] = bool(
      differs and verdict.get("target") and verdict.get("derivation"))
  verdict["model_disagreement"] = verdict["candidate_disagreement"]
  verdict["revision_eligible"] = bool(verdict["grounded"])
  if verdict["revision_eligible"]:
    verdict["grounding_status"] = "eligible"
  elif verdict["candidate_disagreement"]:
    verdict["grounding_status"] = "candidate"
  else:
    verdict["grounding_status"] = "unverified"
  if verdict["grounded"]:
    verdict["decision"] = "disagree" if differs else "agree"
    verdict["overall"] = "flawed" if differs else "ok"
  else:
    verdict["decision"] = "unverified"
    verdict["overall"] = "ok"
  return verdict


def critique_solver(question, solver_response, verdict, tool_calls, tracker, run_id, cfg):
  """Show the solver's work only after the auditor has completed its independent solve."""
  checks = "\n".join(
      f"- {call['tool']}: {call['input']}\n  result: {call['output']}"
      for call in tool_calls) or "No tools were used."
  fields = {
      "question": question,
      "solver_response": solver_response,
      "auditor_answer": verdict.get("auditor_answer", "") or "No answer",
      "interpretation": verdict.get("interpretation", "") or "Not established",
      "derivation": verdict.get("derivation", "") or "No derivation",
      "tool_checks": checks,
  }
  prompt = re.sub(
      r"\{(question|solver_response|auditor_answer|interpretation|derivation|tool_checks)\}",
      lambda match: fields[match.group(1)], AUDITOR_CRITIQUE_PROMPT)
  return call_llm(
      tracker, model=cfg.model,
      system="You are a careful math auditor. Critique only what the evidence supports.",
      messages=[{"role": "user", "content": prompt}],
      temperature=cfg.temperature, max_tokens=cfg.max_tokens,
      run_id=run_id, tag=f"critique-{cfg.tag()}")


def run_audit(question, answer, tracker, run_id, cfg=None):
  """Audit one answer. Returns a dict:
      verdict     parsed verdict, or None if the auditor failed
      failure     None, or why it failed (invalid JSON, no verdict, ...)
      steps       number of auditor turns used
      tool_calls  [{tool, input, output}]
  """
  cfg = cfg or AuditorConfig()
  system = build_auditor_system(cfg.style, cfg.use_tools, False, available_modules())
  messages = [{"role": "user", "content": build_auditor_user(question, cfg.max_steps)}]
  tool_calls, verdict, failure, steps = [], None, None, 0
  safe_stop = False
  seen_calls = {}
  tool_error_streak = 0

  for step in range(1, cfg.max_steps + 1):
    steps = step
    last = step == cfg.max_steps
    text = call_llm(tracker, model=cfg.model, system=system, messages=messages,
                    temperature=cfg.temperature, max_tokens=cfg.max_tokens, run_id=run_id,
                    tag=f"audit-{cfg.tag()}-s{step}")
    obj = _extract_json(text)
    if not isinstance(obj, dict) or "action" not in obj:
      failure = "auditor returned invalid JSON"
      break
    messages.append({"role": "assistant", "content": text})
    action = obj.get("action")

    if action == "final":
      verdict = parse_verdict(obj)
      break
    if action not in ("calculator", "python") or not cfg.use_tools or last:
      failure = f"auditor chose an unusable action: {action!r}"
      break
    call_key = (action, str(obj.get("input", "")).strip())
    preflight = _tool_error(action, obj.get("input", ""), seen_calls)
    if preflight:
      out = preflight
    else:
      out = run_tool(action, obj.get("input", ""))
      seen_calls[call_key] = out
    tool_calls.append({"tool": action, "input": str(obj.get("input", ""))[:600], "output": out})
    tool_calls[-1]["echo"] = False
    if out.startswith("ERROR") or out.startswith("(no output"):
      tool_error_streak += 1
    else:
      tool_error_streak = 0
    messages.append({"role": "user", "content": AUDITOR_TOOL_RESULT
                      .replace("{tool}", action).replace("{out}", out)
                      .replace("{next}", AUDITOR_NEXT_LAST if step + 1 == cfg.max_steps else AUDITOR_NEXT)})
    if tool_error_streak >= 2:
      verdict = {
        "claims": [], "overall": "ok", "model_overall": "ok",
        "auditor_answer": "", "candidate_answer": "",
        "target": str(obj.get("target", "")).strip()[:150],
        "problem_type": str(obj.get("problem_type", "")).strip()[:80],
        "interpretation": "", "independence": "", "derivation": "",
        "verification_plan": "", "confidence": "low",
        "evidence_backed": False, "approximate_evidence": False,
        "decision": "unverified", "grounded": False, "grounding_checks": 0,
        "method_diverse": False, "revision_eligible": False,
      }
      failure = "stopped after repeated malformed tool calls"
      safe_stop = True
      break

  if verdict is None and failure is None:
    failure = "no verdict"
  if verdict is not None:
    decide(verdict, answer, tool_calls)
  return {"verdict": verdict, "failure": failure, "safe_stop": safe_stop,
          "steps": steps, "tool_calls": tool_calls}