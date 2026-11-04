"""Every prompt string in the project lives here, so wording is changed in one place.

SOLVE_PROMPT is used by the pilot tagging AND by every system's first answer.
Keeping them identical is what makes the pilot tags describe the same solver
that is later evaluated.
"""

SOLVE_PROMPT = (
    "Solve the following math problem. Think step by step, but keep your reasoning concise: "
    "show only the essential steps.\n\n"
    "Problem:\n{q}\n\n"
    "Finish with a single final line in exactly this form, with nothing after it:\n"
    "ANSWER: <final answer only, as a number or simple fraction, no units>"
)