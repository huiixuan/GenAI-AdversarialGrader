# GenAI-AdversarialGrader

This project compares four LLM-based systems for grading math answers, ranging
from a single-shot solver to an adaptive auditor that only verifies answers when
it detects uncertainty. All models run locally through Ollama — no external API
or cost required.

---

## Quick Start

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Install Ollama and pull the models**

Download Ollama from https://ollama.com/download, then:
```bash
ollama pull llama3.2:3b   # solver
ollama pull llama3.1:8b   # auditor
ollama pull qwen2.5:7b    # third model for comparison
```

**3. Run a system on the dev set**
```bash
# System 1 — Baseline
python -m systems.baseline --split dev --run-id baseline_dev

# System 2 — Self-Reflection
python -m systems.self_reflection --split dev --run-id compare_dev

# System 3 — Adversarial Grader
python -m scripts.eval_adversarial --split dev --run-id adversarial_dev

# System 4 — Adaptive Auditor
python3 systems/adaptive_auditor.py --split dev
```

Results are saved to `results/` automatically.

> **Note:** `dev.json` is for development and tuning — run it as many times as
> you like. `test.json` is sealed — run each system once only with a fixed
> `--run-id`. Tuning after seeing test results invalidates the numbers.

---

## Repository Layout

```
data/               Question sets with pilot tags
  all.json            243 questions (full set)
  dev.json             73 questions (30% — for tuning)
  test.json           170 questions (70% — sealed, run once)

systems/            One file per grading system
  baseline.py         System 1
  self_reflection.py  System 2
  auditor.py          System 3 auditor logic
  adversarial.py      System 3 pipeline
  adaptive_auditor.py System 4

scripts/
  prepare_data.py     Builds the dataset from Hugging Face
  pilot_tag.py        Tags questions by solver pass rate
  eval_adversarial.py Runs System 3 evaluation

results/            Output from each run (created automatically)

grading.py          Answer extraction and numeric equivalence checking
llm.py              Ollama calls with caching and cost tracking
prompts.py          All prompt strings in one place
config.py           Model names, temperatures, thresholds
tools.py            Calculator and Python tool used by the auditor
```

---

## Dataset

Questions come from two public benchmarks:

- **GSM8K** — grade-school word problems (99 questions)
- **MATH** — competition math, levels 2–5, seven subjects (144 questions)

Only questions with numeric or simple-fraction answers are included. The dataset
is already built and committed — you do not need to run the data preparation
scripts unless you want to rebuild from scratch.

Each question has a **pilot tag** assigned by running the solver five times and
checking how often it gets the right answer:

| Tag | Meaning | Count |
|---|---|---|
| `consistent_correct` | Solver always correct | 81 |
| `seeded_error` | Solver sometimes right, sometimes wrong | 105 |
| `always_wrong` | Solver always wrong | 57 |

The `seeded_error` questions are the main target — these are where an auditor can
actually make a difference. When reporting results, always report on the full set
and the `seeded_error` subset separately.

---

## System Architecture

All four systems start from the same first solver call. What differs is what
happens after:

```
Question (dev.json / test.json)
            │
            ▼
  ┌─────────────────────┐
  │  Solver llama3.2:3b │  ← shared first call across all systems
  │     SOLVE_PROMPT    │
  └──────────┬──────────┘
             │
     ┌───────┴──────────────────────────────────┐
     │               │              │            │
System 1         System 2       System 3      System 4
(Baseline)   (Self-Reflection) (Adversarial  (Adaptive
     │               │            Grader)      Auditor)
     │          Critique +            │             │
     │          Revise (×1)    Blind Solve    Risk Score
     │               │         + Audit        Routing
     │               │         + Revise           │
     ▼               ▼              ▼              ▼
Final Answer   Final Answer   Final Answer   Final Answer
```

---

## Running Each System

### System 1 — Baseline

One solver call per question, no verification.

```bash
# Dev (run freely)
python -m systems.baseline --split dev --run-id baseline_dev

# Single model only
python -m systems.baseline --split dev --run-id baseline_dev --models llama3.2:3b

# Test (run ONCE only)
python -m systems.baseline --split test --run-id baseline_test_final --final
```

### System 2 — Self-Reflection

Solver answers, critiques, then revises. Use the same `--run-id` as the baseline
to reuse the cached first answer.

```bash
# Dev
python -m systems.baseline --split dev --run-id compare_dev
python -m systems.self_reflection --split dev --run-id compare_dev

# Test (run ONCE only)
python -m systems.self_reflection --split test --run-id reflect_test_final --final
```

### System 3 — Adversarial Grader

An independent auditor (llama3.1:8b) solves the question blindly, then critiques
the solver's answer. The solver revises if the answers disagree. The auditor can
use a calculator or Python tool to verify calculations.

```bash
# Dev
python -m scripts.eval_adversarial --split dev --run-id adversarial_dev

# Dev without tools (ablation)
python -m scripts.eval_adversarial --split dev --run-id adversarial_dev_notools --no-tools

# Test (run ONCE only)
python -m scripts.eval_adversarial --split test --run-id adversarial_test_final --final
```

### System 4 — Adaptive Auditor

Samples the solver multiple times, computes a risk score from answer disagreement,
and only calls the auditor when risk is high enough to justify the cost.

```bash
# Dev (tune knobs here freely)
python3 systems/adaptive_auditor.py --split dev

# Test (run ONCE only)
python3 systems/adaptive_auditor.py --split test --run-id adaptive_test_final
```

**Tuning knobs** — edit the top of `systems/adaptive_auditor.py`:

| Knob | Default | What it does |
|---|---|---|
| `SAMPLE_K_INITIAL` | 3 | Solver calls before first risk check |
| `SAMPLE_K_EXTRA` | 2 | Extra calls for borderline risk zone |
| `RISK_LOW` | 0.3 | Below this: trust and stop |
| `RISK_HIGH` | 0.5 | Above this: audit immediately |
| `RISK_THRESHOLD` | 0.4 | Audit trigger after full sample set |

---

## Results

All results are saved to `results/` after each run:

| File | Contents |
|---|---|
| `<run_id>_<split>_baseline-<model>.jsonl` | Per-question answers, System 1 |
| `<run_id>_<split>_self-reflection-<model>.jsonl` | Per-question answers, System 2 |
| `<run_id>_metrics.json` | Aggregate metrics, System 4 |
| `<run_id>_results.json` | Per-question results, System 4 |
| `llm_cache.jsonl` | Cached LLM responses (shared across systems) |

Re-running with the same `--run-id` reuses cached LLM responses. A changed
prompt or model automatically creates a new cache entry.

---

## Answer Checking

The grader (`grading.py`) looks for a final `ANSWER: <value>` line, then falls
back to "the answer is …" or a standalone number. Numeric answers are compared
as exact fractions, so `1/2` and `0.5` are treated as equal.
