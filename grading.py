"""Answer extraction, normalisation and grading. No project imports:
this file sits at the bottom of the dependency chain."""
import re
from fractions import Fraction


def last_boxed(text):
  """Return the content of the last \\boxed{...} (handles nested braces)."""
  idx = text.rfind("\\boxed")
  if idx < 0:
    return None
  i = text.find("{", idx)
  if i < 0:
    return None
  depth = 0
  for j in range(i, len(text)):
    if text[j] == "{":
      depth += 1
    elif text[j] == "}":
      depth -= 1
      if depth == 0:
        return text[i + 1:j]
  return None


_LENIENT_PATTERNS = [
  r"(?i)final\s+answer\s*(?:is)?\s*[:=]\s*([^\n]+)",
  r"(?i)\banswer\s*[:=]\s*([^\n]+)",
  r"(?i)\bthe\s+(?:final\s+|correct\s+|right\s+)?answer\s+is\s*:?\s*([^\n]+)",
]


def _clean_answer(s):
  """Strip markdown emphasis, \\boxed{}, and a trailing period from an answer string."""
  s = re.sub(r"^[\s*_`]+|[\s*_`]+$", "", s.strip())
  b = last_boxed(s)
  if b is not None:
    return b.strip()
  s = s.rstrip(".").strip()
  m = re.fullmatch(r"[A-Za-z]{1,3}\s*=\s*(.+)", s)      # 't = 0.75' -> '0.75'
  if m:
    s = m.group(1).strip()
  return s or None


_LEADING_NUMBER = re.compile(r"^\s*(-?\$?\s*[\d,]*\.?\d+(?:\s*/\s*\d+)?|\\frac\{\d+\}\{\d+\})(?![\w.])")


def _numeric_candidate(cand):
  """For free-text matches ('the answer is 0, but ...'): keep a plain number, or the number the
  clause starts with. Anything else is rejected so later rules can try, instead of returning prose."""
  if cand is None:
    return None
  if to_fraction(cand) is not None:
    return cand
  m = _LEADING_NUMBER.match(cand)
  return m.group(1).strip() if m and to_fraction(m.group(1)) is not None else None


def _last_line_answer(text):
  """Safety net for models that end on their final computation instead of an ANSWER line.
  Takes the text after the last '=' (else the last ':') of the final line, and accepts it ONLY if
  it is a plain number or simple fraction. A cut-off or free-text line is rejected, not guessed."""
  lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
  if not lines:
    return None
  last = re.sub(r"[*_`]", "", lines[-1]).strip().rstrip(".").strip()
  cand = last
  for sep in ("=", ":"):
    if sep in last:
      cand = last.rsplit(sep, 1)[1].strip().rstrip(".").strip()
      break
  cand = _clean_answer(cand) if cand else None
  return cand if cand and to_fraction(cand) is not None else None


def extract_with_mode(text):
  """Pull the final answer out of a model response. Returns (answer, mode), where mode records
  HOW it was found so you can report it:
    'marker'    the last 'ANSWER: x' line (the format we ask for)
    'lenient'   'Answer: x', '**Final Answer:** x', 'the (correct) answer is x'
    'boxed'     the last \\boxed{x}
    'last_line' no marker at all: the numeric result on the final line (see _last_line_answer)
    None        nothing usable: counted as wrong and tallied as 'unparsable'
  """
  if not text:
    return None, None
  strict = re.findall(r"ANSWER:\s*(.+)", text)
  if strict:
    a = _clean_answer(strict[-1])
    if a:
      return a, "marker"
  found = []
  for pat in _LENIENT_PATTERNS:
    for m in re.finditer(pat, text):
      found.append((m.start(1), m.group(1)))
  for _, raw in sorted(found, reverse=True):          # latest match first
    a = _numeric_candidate(_clean_answer(raw))
    if a:
      return a, "lenient"
  b = last_boxed(text)
  if b is not None:
    return b, "boxed"
  a = _last_line_answer(text)
  if a is not None:
    return a, "last_line"
  return None, None


def extract_model_answer(text):
  """Answer only (see extract_with_mode)."""
  return extract_with_mode(text)[0]


def normalize(ans):
  """Strip formatting noise so '\\dfrac{1}{2}', '1/2', ' 1/2. ' look alike."""
  if ans is None:
    return ""
  s = str(ans).strip()
  s = re.sub(r"\\text\{\s*([^}]*)\}", r"\1", s)        # \text{cm} -> cm
  s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
  s = s.replace("\\left", "").replace("\\right", "")
  s = s.replace("\\!", "").replace("\\,", "").replace("\\ ", "")
  s = s.replace("\\$", "").replace("$", "")
  s = s.replace("\\%", "").replace("%", "")
  s = s.replace("^\\circ", "").replace("^{\\circ}", "").replace("°", "")
  s = s.replace(",", "") if re.fullmatch(r"-?[\d,]+(\.\d+)?", s.replace(" ", "")) else s
  s = s.strip().rstrip(".")
  # drop trailing units/words after a number: "42 cm" -> "42"
  m = re.fullmatch(r"(-?[\d.,/\\{}a-z]+?)\s+[A-Za-z][A-Za-z\s\.\^\d]*", s)
  if m and re.search(r"\d", m.group(1)):
    s = m.group(1)
  s = s.replace(" ", "")
  # \frac12 style
  s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
  return s


def to_fraction(s):
  """Parse a normalised string to a Fraction, or None if not a plain number."""
  s = normalize(s)
  if not s:
    return None
  sign = 1
  if s.startswith("-"):
    sign, s = -1, s[1:]
  elif s.startswith("+"):
    s = s[1:]
  m = re.fullmatch(r"\\frac\{(\d+)\}\{(\d+)\}", s)
  if m and int(m.group(2)) != 0:
    return sign * Fraction(int(m.group(1)), int(m.group(2)))
  m = re.fullmatch(r"(\d+)/(\d+)", s)
  if m and int(m.group(2)) != 0:
    return sign * Fraction(int(m.group(1)), int(m.group(2)))
  if re.fullmatch(r"\d+(\.\d+)?", s):
    return sign * Fraction(s)
  return None


def is_numeric_gold(ans):
  return to_fraction(ans) is not None


def is_correct(pred, gold):
  """Exact match after normalisation; numeric answers compared as Fractions."""
  if pred is None:
    return False
  fp, fg = to_fraction(pred), to_fraction(gold)
  if fp is not None and fg is not None:
    return fp == fg
  return normalize(pred) == normalize(gold)


def selftest():
  cases = [
    ("1/2", "\\frac{1}{2}", True), ("0.5", "\\frac{1}{2}", True),
    ("\\dfrac{1}{2}", "1/2", True), ("1,234", "1234", True),
    ("-3", "-3.0", True), ("42 cm", "42", True), ("\\$18", "18", True),
    ("50\\%", "50", True), ("90^\\circ", "90", True),
    ("\\text{12}", "12", True), ("126.", "126", True),
    ("2/3", "0.667", False), ("5", "6", False), (None, "5", False),
    ("\\frac12", "\\frac{1}{2}", True), ("-\\frac{3}{4}", "-3/4", True),
  ]
  bad = 0
  for pred, gold, want in cases:
    got = is_correct(pred, gold)
    if got != want:
      bad += 1
      print(f"FAIL  pred={pred!r} gold={gold!r} expected={want} got={got}")
  assert last_boxed("so \\boxed{\\frac{1}{2}} done") == "\\frac{1}{2}"
  assert extract_model_answer("blah\nANSWER: 17") == "17"
  assert extract_model_answer("x \\boxed{9}") == "9"
  assert extract_model_answer("done\nAnswer: **12**") == "12"
  assert extract_model_answer("**Final Answer:** 7.") == "7"
  assert extract_model_answer("So the answer is 15.") == "15"
  assert extract_model_answer("The final answer is: $\\boxed{5}$") == "5"
  assert extract_model_answer("ANSWER: **5**") == "5"
  assert extract_model_answer("I ran out of space while") is None
  assert extract_model_answer("the answer is 0, but I can write the correct number:\n\n0") == "0"
  assert is_correct(extract_model_answer("so the answer is 5 apples."), "5")
  assert is_correct(extract_model_answer("Red left: 9 - 3 = 6\nTotal: 4 + 6 = 10"), "10")
  assert is_correct(extract_model_answer("Step 3: so the final amount is: $90"), "90")
  assert extract_with_mode("a\nTotal: 4 + 6 = 10")[1] == "last_line"
  assert extract_with_mode("a\nANSWER: 4")[1] == "marker"
  assert extract_model_answer("This is a contradiction.\n\nANSWER:") is None
  assert extract_model_answer("x = 2^10 * 3^2 * 5^") is None
  assert extract_model_answer("I think the result is large") is None
  assert extract_model_answer("1. x*(2x+1) = 2*(109") is None
  assert extract_model_answer("so the correct answer is:\n\nt = 0.75") == "0.75"
  assert extract_model_answer("ANSWER: x = 5") == "5"
  print(f"grader selftest: {len(cases) - bad}/{len(cases)} passed")
  if bad:
    raise SystemExit(1)


if __name__ == "__main__":
  selftest()