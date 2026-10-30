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


def extract_model_answer(text):
  """Pull the final answer out of a model response.
  Priority: 'ANSWER: x' line -> \\boxed{x} -> None."""
  if not text:
    return None
  m = re.findall(r"ANSWER:\s*(.+)", text)
  if m:
    ans = m[-1].strip()
    b = last_boxed(ans)
    return b if b is not None else ans
  return last_boxed(text)


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
  print(f"grader selftest: {len(cases) - bad}/{len(cases)} passed")
  if bad:
    raise SystemExit(1)


if __name__ == "__main__":
  selftest()