# GenAI-AdversarialGrader

The Flow
Hugging Face datasets
        |
        v
scripts/prepare_data.py
  - loads GSM8K and MATH
  - filters out unsuitable questions and non-numeric answers
  - removes duplicates
  - samples questions and creates dev/test splits
        |
        v
data/all.json  ------> data/dev.json
        |               data/test.json
        v
scripts/pilot_tag.py  <---- local Ollama model (llama3.2:3b)
  - asks the model to solve each question k times
  - parses each response and checks it against the gold answer
  - labels each question based on its pass rate
        |
        +----> results/llm_cache.jsonl
        |        cached model responses and usage/latency metadata
        |
        +----> data/review_always_wrong.json
        |        questions that the model got wrong every time, for review
        v
manually review questionable questions
        |
        v
data/exclude_ids.txt ---> apply exclusions with --exclude


The answer parser and checker live in grading.py; 
model calls and caching are handled by llm.py. 
The prompt used for solving is in prompts.py.


What the pilot tags mean
For each question, the pilot makes five attempts by default. It compares each extracted answer with the gold answer, then assigns a tag:

consistent_correct: all attempts were correct.
seeded_error: some attempts were correct and some wrong.
always_wrong: all attempts were wrong; these are collected for manual review.
The solver prompt asks for a final line like ANSWER: 42. The grader tries that format first, then uses fallbacks such as “the answer is …” or a final numeric line. Numeric answers are compared as exact fractions where possible, so equivalent forms like 1/2 and 0.5 match.


What’s already in this checkout
The dataset files already exist. data/all.json contains 243 questions with pilot predictions and tags; data/dev.json has 73 and data/test.json has 170. The current tags are 81 consistent_correct, 105 seeded_error, and 57 always_wrong.

The tagged results/llm_cache.jsonl has 26 entries. Each entry is a cached model response with token counts and latency. It is not a full evaluation report; for example, the sample records show different attempts answering the same questions.

Systems
-------

System 4 — Adaptive Auditor (systems/adaptive_auditor.py)

The adaptive auditor improves on a single-shot solver by only paying for a
second "audit" call when the model’s own answers are inconsistent.

Pipeline (per question):

  1. SOLVE   — ask the model once with SOLVE_PROMPT
  2. SAMPLE  — draw (SAMPLE_K - 1) more answers at solver temperature
  3. RISK    — risk = 1 - (modal_count / SAMPLE_K)
               0.0 = all samples agree  -> low risk, trust majority vote
               1.0 = maximally split    -> high risk, send to auditor
  4. AUDIT   — if risk >= RISK_THRESHOLD:
               send question + modal answer to an auditor LLM (temp=0)
               auditor checks the working and may override the answer
  5. VERDICT — final answer = auditor answer (if triggered) else majority vote

Tuning knobs (top of adaptive_auditor.py, tune on dev.json only):

  SAMPLE_K         default 5    Total solver samples; higher = better risk signal, more cost
  RISK_THRESHOLD   default 0.4  Audit when risk >= this; lower = more audits triggered
  AUDIT_PROMPT                  The auditor instruction; edit wording to improve accuracy

Metrics reported:
  - Overall accuracy and audit trigger rate
  - Accuracy broken down by tag (seeded_error, consistent_correct, always_wrong)
  - Auditor targeting: precision and recall against seeded_error, measuring
    whether the risk signal correctly routes adversarial questions to the auditor

Running:

  # Tune on dev -- run as many times as you like
  python3 systems/adaptive_auditor.py --split dev

  # Final reported result -- run ONCE only, do not tune after seeing this
  python3 systems/adaptive_auditor.py --split test --run-id adaptive_test_final

Results are saved to results/:
  <run_id>_metrics.json    aggregate metrics
  <run_id>_results.json    per-question predictions, risk scores, audit flags

Why adaptive?
seeded_error questions are inconsistent by construction (pilot pass rate 0.2-0.8),
so they naturally produce higher risk scores. The auditor therefore fires
preferentially on adversarial cases, giving better accuracy than a plain solver
at lower cost than an always-on second pass.


Other systems
-------------

systems/baseline.py — single-shot solver, no checking (in progress).
