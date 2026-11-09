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

System 1 — Baseline (systems/baseline.py)

The baseline is one LLM call. It solves the question with SOLVE_PROMPT and returns
that answer. initial_answer and final_answer are the same parsed answer.

The call uses the same prompt, temperature, and tag ("solve") as every other
system's first answer. On llama3.2:3b, which is config.SOLVER_MODEL, a shared
--run-id reuses the cached first answer already stored for the other systems.
Each other model is its own cache entry.

Default models, all local through Ollama:

  llama3.2:3b    solver in config.py
  llama3.1:8b    auditor in config.py
  qwen2.5:7b     a third open model

Pull them once before the first run. The script does not download weights.
If a model is missing, it stops and prints the pull command.

  ollama pull llama3.2:3b
  ollama pull llama3.1:8b
  ollama pull qwen2.5:7b

Running (from the project root):

  # all three models on dev
  python -m systems.baseline --split dev --run-id baseline_dev

  # one model, or a short prefix of the shuffled dev split
  python -m systems.baseline --split dev --run-id baseline_dev --models llama3.2:3b --limit 20

  # reprint the report from saved results
  python -m systems.baseline --report --split dev --run-id baseline_dev

  # sealed test split — run ONCE only
  python -m systems.baseline --split test --run-id baseline_test_final --final

Pass a comma-separated list to --models to change the set. --report reads the
same list, so name any custom model again when reprinting its results.

A changed prompt, temperature, or model refuses to resume an existing file.
Use a new --run-id after any of those changes. The test split needs --final.

Results are saved to results/. A colon in a model name is written as a hyphen,
so llama3.2:3b becomes llama3.2-3b:

  <run_id>_<split>_baseline-<model>.jsonl           per-question answers and cost
  <run_id>_<split>_baseline-<model>_metrics.json    accuracy, unparsed rate, cost

Metrics reported:

  - Overall accuracy and the share of answers with no parsable final line
  - Accuracy by tag (seeded_error, consistent_correct, always_wrong)
  - Mean calls, tokens, and latency per question
  - A comparison table when more than one model was run


System 2 — Self-reflection (systems/self_reflection.py)

The same model answers, critiques that answer, and then revises it. There is
no separate auditor and no tools.

Pipeline (per question):

  1. SOLVE    — one SOLVE_PROMPT call, the same first answer as the baseline
  2. CRITIQUE — the same model reviews its own solution and names a specific mistake,
                or says the solution is correct
  3. REVISE   — the same model solves the problem again in light of that critique
                and ends with a new ANSWER line

All three calls use SOLVER_TEMP and the model selected for that run. The first
call uses tag "solve", so it is the cached baseline answer when this system is
run with the same --run-id. Critique and revision are extra calls. If the
critique is empty, or the revision has no parsable answer, the first answer
is kept.

The default models are the same three as the baseline. Pull them once:

  ollama pull llama3.2:3b
  ollama pull llama3.1:8b
  ollama pull qwen2.5:7b

Running (from the project root):

  # same run-id as the baseline, so the first answer is shared
  python -m systems.baseline --split dev --run-id compare_dev
  python -m systems.self_reflection --split dev --run-id compare_dev

  # one model, or a short prefix of the shuffled dev split
  python -m systems.self_reflection --split dev --run-id compare_dev --models llama3.2:3b --limit 20

  # reprint the report from saved results
  python -m systems.self_reflection --report --split dev --run-id compare_dev

  # sealed test split — run ONCE only
  python -m systems.self_reflection --split test --run-id reflect_test_final --final

A changed prompt, temperature, or model refuses to resume an existing file.
Use a new --run-id after any of those changes. The test split needs --final.

Results are saved to results/. A colon in a model name is written as a hyphen:

  <run_id>_<split>_self-reflection-<model>.jsonl           per-question answers, critique, cost
  <run_id>_<split>_self-reflection-<model>_metrics.json    first vs final accuracy, cost

Metrics reported:

  - Accuracy of the first answer and of the revised answer
  - How often the answer changed, how many wrong answers became right, and how many right answers became wrong
  - The same counts by tag (seeded_error, consistent_correct, always_wrong)
  - Mean calls, tokens, and latency per question
  - A comparison table when more than one model was run


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
