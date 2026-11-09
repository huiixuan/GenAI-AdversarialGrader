"""Tools the auditor can call.

calculator(expr)  exact arithmetic on a safe subset of Python syntax (no names, no attributes).
run_python(code)  short Python programs in a separate process with a timeout.

SAFETY NOTE: run_python is a best-effort guard against a confused model, NOT a security boundary
against a malicious one. Code is checked against an allowlist of imports and a few banned names,
then run in a fresh interpreter (isolated mode, empty environment, temp working directory, CPU and
time limits). Don't expose it to untrusted users.
"""
import ast
import math
import importlib.util
import operator
import os
import re
import subprocess
import sys
import tempfile

from config import CODE_TIMEOUT_S, TOOL_OUTPUT_CHARS
from grading import to_fraction

AUDITOR_ANSWER_MARKER = "AUDITOR_ANSWER:"
_AUDITOR_ANSWER_RE = re.compile(
    rf"^{re.escape(AUDITOR_ANSWER_MARKER)}\s*"
    r"(-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?|\\frac\{\d+\}\{\d+\})\s*$"
)


def extract_marked_answer(output):
    """Return the sole final marked answer, or None if the output is not evidence."""
    lines = [line.strip() for line in str(output).splitlines() if line.strip()]
    if not lines:
        return None
    matches = [_AUDITOR_ANSWER_RE.fullmatch(line) for line in lines]
    if sum(match is not None for match in matches) != 1 or matches[-1] is None:
        return None
    return matches[-1].group(1)


def is_echo_input(action, input_text, solver_answer):
    """Return whether a tool request merely repeats the solver's claimed answer."""
    target = to_fraction(solver_answer) if solver_answer is not None else None
    if target is None:
        return False
    text = str(input_text).strip()
    if not text:
        return False
    if action == "calculator":
        return to_fraction(text.rstrip(";")) == target
    if action != "python":
        return False
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return False
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Expr):
        return False
    call = tree.body[0].value
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) \
            or call.func.id != "print":
        return False
    argument_nodes = [node for arg in call.args for node in ast.walk(arg)]
    argument_nodes += [node for keyword in call.keywords for node in ast.walk(keyword.value)]
    if any(isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Name, ast.Attribute,
                             ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign))
           for node in argument_nodes):
        return False
    return any(to_fraction(node.value) == target for node in argument_nodes
               if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str)))


def _clip(s):
    s = str(s)
    return s if len(s) <= TOOL_OUTPUT_CHARS else s[:TOOL_OUTPUT_CHARS] + f"... [truncated, {len(s)} chars]"


# --------------------------------------------------------------------------
# calculator
# --------------------------------------------------------------------------
_BIN = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
_FUNCS = {"sqrt": math.sqrt, "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil,
          "factorial": math.factorial, "comb": math.comb, "perm": math.perm, "gcd": math.gcd,
          "lcm": math.lcm, "log": math.log, "log10": math.log10, "log2": math.log2,
          "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan,
          "min": min, "max": max}
_NAMES = {"pi": math.pi, "e": math.e}
_CMP = {ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
        ast.Gt: operator.gt, ast.GtE: operator.ge}


def _pow(a, b):
    if abs(b) > 10_000:
        raise ValueError("exponent too large")
    if isinstance(a, int) and isinstance(b, int) and b > 0 and abs(a) > 1 \
            and b * math.log10(abs(a)) > 20_000:
        raise ValueError("result too large")
    return a ** b


def _eval(node):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp):
        l, r = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            return _pow(l, r)
        if type(node.op) in _BIN:
            return _BIN[type(node.op)](l, r)
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _CMP:
        return _CMP[type(node.ops[0])](_eval(node.left), _eval(node.comparators[0]))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _eval(node.operand)
        return +v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.Name) and node.id in _NAMES:
        return _NAMES[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
            and node.func.id in _FUNCS and not node.keywords:
        args = [_eval(a) for a in node.args]
        if node.func.id == "factorial" and args and args[0] > 2000:
            raise ValueError("factorial argument too large")
        return _FUNCS[node.func.id](*args)
    raise ValueError("unsupported syntax")


def _fmt_number(x):
    if isinstance(x, float):
        if x == int(x) and abs(x) < 1e15:
            return str(int(x))
        return repr(float("%.12g" % x))
    return str(x)


CALC_HELP = ("The calculator takes ONE expression with numbers and operators (+ - * / ** % //), "
             "functions like sqrt(), comb(), gcd(), and comparisons like 105 % 7 == 1. "
             "For equations, algebra, lists or loops, use the python tool.")


def calculator(expr):
    """Evaluate one arithmetic expression. '^' is read as a power, as people write it."""
    try:
        e = str(expr).strip().rstrip("=").replace("^", "**").replace("×", "*").replace("÷", "/")
        if not e:
            return "ERROR: empty expression. " + CALC_HELP
        if re.search(r"(?<![=!<>])=(?!=)", e):
            return "ERROR: a single '=' is not allowed; use '==' to compare. " + CALC_HELP
        value = _fmt_number(_eval(ast.parse(e, mode="eval")))
        return _clip(f"{AUDITOR_ANSWER_MARKER} {value}")
    except ZeroDivisionError:
        return "ERROR: division by zero"
    except SyntaxError:
        return "ERROR: not a valid arithmetic expression. " + CALC_HELP
    except Exception as ex:
        return f"ERROR: {ex}. " + CALC_HELP


# --------------------------------------------------------------------------
# sandboxed python
# --------------------------------------------------------------------------
ALLOWED_IMPORTS = {"math", "cmath", "fractions", "decimal", "itertools", "functools", "collections",
                   "statistics", "random", "re", "string", "operator", "bisect", "heapq",
                   "sympy", "numpy"}
BANNED_NAMES = {"open", "exec", "eval", "compile", "input", "breakpoint", "exit", "quit", "globals",
                "locals", "vars", "getattr", "setattr", "delattr", "help", "memoryview"}


def check_code(code):
    """Return None if the code passes the allowlist, else a short reason."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"import of '{a.name.split('.')[0]}' is not allowed"
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level or root not in ALLOWED_IMPORTS:
                return f"import from '{node.module}' is not allowed"
        elif isinstance(node, ast.Name) and (node.id in BANNED_NAMES or node.id.startswith("__")):
            return f"use of '{node.id}' is not allowed"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return "attribute access starting with '_' is not allowed"
    return None


def available_modules():
    """Allowed modules that are actually installed, so the prompt only promises what exists."""
    return [m for m in ("math", "fractions", "itertools", "sympy", "numpy")
            if importlib.util.find_spec(m) is not None]


_PREAMBLE = ("import resource as _r\n"
             "try:\n"
             "    _r.setrlimit(_r.RLIMIT_CPU, ({cpu}, {cpu}))\n"
             "    _r.setrlimit(_r.RLIMIT_AS, (2 << 30, 2 << 30))\n"   # ignored on some platforms (macOS)
             "except Exception:\n"
             "    pass\n"
             "del _r\n")


def run_python(code, timeout=None):
    """Run a short program; return what it printed (or an error message)."""
    timeout = timeout or CODE_TIMEOUT_S
    if not str(code).strip():
        return ("ERROR: empty Python input. Write a complete program with real line breaks and "
                "one statement per line, for example:\n"
                "from fractions import Fraction\n"
                "result = Fraction(3, 32)\n"
                f"print('{AUDITOR_ANSWER_MARKER} ' + str(result))")
    bad = check_code(str(code))
    if bad and bad.startswith("SyntaxError"):
        return (f"ERROR: {bad}. Write each statement on its own line (use real line breaks); a while, "
                "for or if block cannot follow a semicolon.")
    if bad:
        return f"ERROR (blocked): {bad}"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "main.py")
        with open(path, "w") as f:
            f.write(_PREAMBLE.format(cpu=int(timeout) + 1) + str(code))
        try:
            p = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True,
                               timeout=timeout, cwd=d, env={"PYTHONHASHSEED": "0"})
        except subprocess.TimeoutExpired:
            return f"ERROR: timed out after {timeout}s"
        except Exception as ex:
            return f"ERROR: could not run code ({ex})"
    out = (p.stdout or "").strip()
    if p.returncode != 0:
        last = (p.stderr or "").strip().splitlines()[-1:] or ["unknown error"]
        hint = (f" (not installed. Available: {', '.join(available_modules())})"
                if "No module named" in last[0] else "")
        return _clip(f"ERROR: {last[0]}{hint}" + (f"\n(output before the error: {out})" if out else ""))
    if not out:
        return (f"(no output: print exactly one final value as "
                f"`{AUDITOR_ANSWER_MARKER} <value>`)")
    if extract_marked_answer(out) is None:
        return (f"ERROR: Python output must include exactly one final line in the form "
                f"`{AUDITOR_ANSWER_MARKER} <number-or-simple-fraction>`. "
                f"Use a quoted string, for example `print('{AUDITOR_ANSWER_MARKER} ' + str(result))`.")
    return _clip(out)


TOOLS = {"calculator": calculator, "python": run_python}


def run_tool(name, arg):
    fn = TOOLS.get(name)
    return fn(arg) if fn else f"ERROR: unknown tool '{name}'"