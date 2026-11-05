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

What it doesn’t do yet
Despite the repository name, the checked-in code doesn’t currently implement a full adversarial-grading/evaluation run. systems/baseline.py is empty, and there’s no script that runs a grader against the test set and reports evaluation metrics. The implemented workflow builds a dataset and uses a solver to characterize its questions.

To inspect the existing pilot summary without making model calls, run this from the repository root:


For a new pilot run, you’ll also need the Python ollama client and an Ollama server running locally with the configured model available. The README.md currently doesn’t document these steps.

