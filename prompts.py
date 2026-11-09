"""Every prompt string in the project lives here, so wording is changed in one place.

SOLVE_PROMPT is used by the pilot tagging AND by every system's first answer.
Keeping them identical is what makes the pilot tags describe the same solver
that is later evaluated.

Layout matters for small models: the output-format instruction goes LAST, because a 3B model tends
to follow whatever it read most recently. (With the format instruction in the middle and a
'be concise' line at the end, ~11% of pilot answers ended without an ANSWER line.)
"""

SOLVE_PROMPT = (
    "Solve the following math problem. Think step by step, but keep your reasoning concise: "
    "show only the essential steps.\n\n"
    "Problem:\n{q}\n\n"
    "Finish with a single final line in exactly this form, with nothing after it:\n"
    "ANSWER: <final answer only, as a number or simple fraction, no units>"
)


# ==========================================================================
# Auditor (systems 3 and 4)
# ==========================================================================
# NOTE: these use .replace() on {placeholders}, never str.format(), because the text contains
# literal JSON braces and problems contain LaTeX braces.

_INTRO_ADVERSARIAL = (
    "You are an independent math auditor. First solve the problem without seeing any other solver's "
    "answer. {reasoning_line} Compute the requested quantity yourself and report your own answer "
    "and derivation. Do not guess; say you are uncertain when you cannot solve it.")
_INTRO_NEUTRAL = (
    "You are an independent math solver. First solve the problem without seeing any other solver's "
    "answer. {reasoning_line} Compute the requested quantity yourself and report your own answer "
    "and derivation. Do not guess; say you are uncertain when you cannot solve it.")

_REASONING_HIDDEN = "You do NOT see the solver's reasoning."
_REASONING_SHOWN = "You are also shown the solver's reasoning; check it step by step."

_METHOD = """How to work:
1. Solve the problem independently before comparing it with any other answer. First identify the exact requested quantity in `target`, and state units, bounds, or indexing in `interpretation`.
2. Classify it in `problem_type`: arithmetic, algebra, geometry, counting, sequence, word_problem, or unknown. Give a concise derivation that supports your own final answer.
3. Use a tool only when it helps verify a calculation. Calculator is for a single numeric expression. Python is for variables, equations, symbolic algebra, or short deterministic programs. Do not call both tools just to satisfy a quota, and do not send your proposed answer alone to a tool.
4. If using Python, use real line breaks and print a final line like `print('AUDITOR_ANSWER: ' + str(result))`. Never claim a tool succeeded when it returned an error.
5. Set confidence to high only when the target and derivation are clear and internally consistent. Use medium or low when uncertain. Do not use sampling, random trials, or rounded floating-point counts as exact evidence for an exact math question.
6. The solver's answer is not available during this stage. Provide only your independently derived answer; comparison is done afterward by the evaluator.
7. `overall` describes only whether you completed your independent solution: use `ok` when you have a supported solution and `flawed` when you could not solve it. It is not a judgment of another answer."""

_TOOLS_BLOCK = """Tools:
- calculator: input is ONE expression with numbers and operators (+ - * / ** % //), functions such as sqrt(), comb(), gcd(), and comparisons such as 105 % 7 == 1. It cannot solve equations. A successful result is returned as `AUDITOR_ANSWER: <value>`.
- python: input is a short Python program. Only what you print is returned. You may import: {modules}. Use real line breaks and one statement per line. Always quote the marker, for example:
  from fractions import Fraction
  result = Fraction(3, 32)
  print('AUDITOR_ANSWER: ' + str(result))
The final line must be exactly the marked answer; a bare `print(AUDITOR_ANSWER: result)` is invalid.
"""
_NO_TOOLS_BLOCK = "You have no tools. Work carefully, and write the exact checks you did in the evidence fields.\n"

_PROTOCOL_TOOLS = """Each turn reply with ONE JSON object and nothing else, with these fields:
  thought: one short sentence on what you are doing
  target: the exact quantity the problem asks for, in a few words
  problem_type: arithmetic, algebra, geometry, counting, sequence, word_problem, or unknown
  interpretation: the relevant terms, units, bounds, or indexing used to define the target
  independence: a second reasoning method if actually used, otherwise an empty string
  derivation: the exact equations, logical steps, bounds, or case analysis leading to the answer
  verification_plan: the tool check used, or why no tool was needed
  confidence: "low", "medium", or "high"
  action: "calculator", "python", or "final"
  input: the expression or code for the tool; an empty string when action is "final"
  auditor_answer: your independently derived answer (not necessarily copied from a tool)
  claims: [] while using tools; in the final response, list any relevant checks and their evidence
  overall: "pending" while using tools; with action "final", "ok" for a completed independent solution or "flawed" if unable to solve
The evaluator compares your answer with the solver's only after your final response. You have a limited number of steps."""
_PROTOCOL_NO_TOOLS = """Reply with ONE JSON object and nothing else, with these fields:
  thought: one short sentence on what you checked
  target: the exact quantity the problem asks for, in a few words
  problem_type: arithmetic, algebra, geometry, counting, sequence, word_problem, or unknown
  interpretation: the relevant terms, units, bounds, or indexing used to define the target
  independence: how independent checks would support the result, or why they were unavailable
  derivation: the exact reasoning used, or an empty string if it could not be completed
  verification_plan: the checks completed or still required
  confidence: "low", "medium", or "high"
  action: always "final"
  input: always an empty string
  auditor_answer: your own answer to the problem if you know it, else ""
  claims: a list of {text, status, evidence}
  overall: "ok" if every claim passed, otherwise "flawed\""""


def build_auditor_system(style="adversarial", use_tools=True, show_reasoning=False, modules=None):
    intro = _INTRO_ADVERSARIAL if style == "adversarial" else _INTRO_NEUTRAL
    intro = intro.replace("{reasoning_line}", _REASONING_SHOWN if show_reasoning else _REASONING_HIDDEN)
    method = _METHOD.replace(
        "{step1}", "Use the tools for every calculation; do not do arithmetic in your head."
        if use_tools else "Do it step by step and carefully.").replace(
        "{check_with}", "with a tool or a concrete test" if use_tools else "with a concrete test")
    tools = _TOOLS_BLOCK.replace("{modules}", ", ".join(modules or ["math", "fractions", "itertools"])) \
        if use_tools else _NO_TOOLS_BLOCK
    protocol = _PROTOCOL_TOOLS if use_tools else _PROTOCOL_NO_TOOLS
    return "\n\n".join([intro, method, tools, protocol])


def build_auditor_user(question, max_steps=4):
    parts = ["Solve this problem independently:\n" + question.strip()]
    parts.append(f"You have at most {max_steps} steps in total (tool calls plus the final verdict). Begin.")
    return "\n\n".join(parts)


AUDITOR_TOOL_RESULT = "Result of {tool}:\n{out}\n\n{next}"
AUDITOR_NEXT = "Use another tool, or give your final verdict."
AUDITOR_NEXT_LAST = "That was your last tool step. Give your final verdict now."

REVISE_PROMPT = (
    "Review the auditor's critique of your first attempt below. The auditor can be wrong, so do not "
    "blindly accept it. Re-solve the original problem, check the exact requested quantity, and use "
    "the critique to find and correct any mistake. If your original answer was right, keep it; if "
    "the critique identifies a valid error, fix your work. Return your final solution.\n\n"
    "Auditor critique: {critique}\n\n"
    "Keep your reasoning concise. Finish with a single final line in exactly this form, with nothing "
    "after it:\nANSWER: <final answer only, as a number or simple fraction, no units>")


AUDITOR_CRITIQUE_PROMPT = """You first solved this problem independently without seeing the solver's work. Now review the solver's original response against your independent solution.

Identify a specific mathematical error if you can justify one, explaining the correction. Check that both solutions address the same quantity, units, and indexing. If the solver appears correct or the evidence is inconclusive, say so plainly instead of inventing a criticism. If the solver response has no clear final answer, explicitly point that out. Give concise, actionable feedback to help the solver make a final attempt. Do not output JSON.

Problem:
{question}

Solver's original response:
{solver_response}

Your independent answer: {auditor_answer}
Your interpretation: {interpretation}
Your derivation: {derivation}
Your tool checks: {tool_checks}"""


def render_verdict(verdict, solver_answer=None):
    """Verdict dict -> the text the solver sees."""
    lines = []
    if verdict.get("decision") == "disagree":
        what = f" as '{verdict['target']}'" if verdict.get("target") else ""
        lines.append(f"The reviewer independently computed {verdict['auditor_answer']}{what} (the calculation is "
                     f"in the evidence below), which differs from your answer {solver_answer}. Check that the "
                     f"reviewer computed the quantity the problem actually asks for.")
    elif verdict.get("decision") == "no_answer":
        lines.append("Your response did not end with a final answer.")
    else:
        lines.append(verdict["overall"].upper())
    if verdict.get("interpretation"):
        lines.append(f"Interpretation used: {verdict['interpretation']}")
    if verdict.get("problem_type"):
        lines.append(f"Problem type: {verdict['problem_type']}")
    if verdict.get("derivation"):
        lines.append(f"Derivation: {verdict['derivation']}")
    if verdict.get("independence"):
        lines.append(f"Independence check: {verdict['independence']}")
    if verdict.get("verification_plan"):
        lines.append(f"Verification plan: {verdict['verification_plan']}")
    for c in verdict["claims"]:
        lines.append(f"- [{c['status'].upper()}] {c['text']}" + (f" ({c['evidence']})" if c["evidence"] else ""))
    return "\n".join(lines)