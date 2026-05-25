# Codex setup spec: isolated translation-routing autoresearch loop for this manga translator repo

You are Codex working inside the manga translator repository root. The local folder may be named `Translator` even if the GitHub repository has a different name. Your task is to set up an isolated autoresearch loop for the next optimization target after rendering: **translation reliability through source-risk classification and translator routing**.

The loop must be separate from the main project. Do **not** put loop scripts, benchmark cases, run outputs, result logs, generated reports, or helper scripts into `.testing/`, `manga_local_translator/`, `quality-runs/`, or any other existing project folder. Everything for this loop must live under one new root folder:

```text
translation_routing_autoresearch/
```

The main project is only an import target and, later, the code under test. The autoresearch harness itself is a separate research system.

---

## 1. Non-negotiable separation rule

Create this folder in the repository root:

```text
translation_routing_autoresearch/
```

All new loop assets must go inside it.

Allowed new files during setup:

```text
translation_routing_autoresearch/**
```

Forbidden setup locations:

```text
.testing/**
manga_local_translator/**
quality-runs/**
.models/**
.model*/**
render_autoresearch/**
root-level helper scripts such as eval_translation_routing.py
root-level result files such as translation_routing_results.tsv
```

The setup phase must not edit main project source code. It may only read project files and create the isolated loop folder.

After setup, future optimization experiments may edit a very small project-code surface, but benchmark fixtures, scoring code, result logs, and run artifacts must remain inside `translation_routing_autoresearch/`.

---

## 2. Purpose of the loop

The loop should optimize **routing and validation decisions for already-OCRed text**, not raw translation model quality.

Given a frozen OCR line or grouped block, the policy under test should decide whether the line should be:

```text
1. skipped as metadata/noise,
2. handled by phrasebook / known-fragment rules,
3. translated with the default local translator,
4. translated with CAT,
5. retried through CAT source-only / strict retry,
6. routed to Qwen critic / Q8 repair,
7. or marked for human review.
```

The benchmark must not run detection, OCR, rendering, inpainting, GUI code, model downloads, Ollama, llama.cpp, Qwen inference, CAT inference, OPUS inference, MADLAD inference, or Argos inference.

The benchmark must use frozen inputs:

```text
source OCR text
neighbor/context text
OCR confidence / geometry metadata
frozen candidate outputs from each translator route
expected risk labels
expected acceptable behavior
```

Lower score is better.

---

## 3. What this loop is not

Do not optimize:

```text
rendering/layout
text detection
OCR model behavior
full-image Qwen vision behavior
font fitting
model download/setup
GUI behavior
benchmark images
translation model prompts that require live model calls
```

Rendering has already been optimized separately. Do not use `render_autoresearch/` for this loop and do not mix the two harnesses.

---

## 4. Project files Codex should inspect before implementation

Read these files to understand the current pipeline, existing guardrails, and available helper functions:

```text
README.md
cat_reliability_improvement_plan.md
translation_quality_improvement_plan.md
manga_translator_hallucination_controls_codex.md
manga_translator_hallucination_controls_codex_advanced.md
translation_glossary.example.json
manga_local_translator/translation_rules.py
manga_local_translator/translation_evidence.py
manga_local_translator/qwen_validation.py
manga_local_translator/text_filter.py
manga_local_translator/translate.py
manga_local_translator/hf_translators.py
manga_local_translator/qwen_translator.py
manga_local_translator/quality_eval.py
manga_local_translator/debug_report.py
manga_local_translator/line_identity.py
.testing/tests/
```

Do not copy large chunks of main-project logic into the harness. Import and call project functions directly where useful.

The harness should tolerate minor function-signature changes. Put all fragile imports behind a small adapter layer in:

```text
translation_routing_autoresearch/scripts/io_adapters.py
```

---

## 5. Required folder tree

Create this structure exactly:

```text
translation_routing_autoresearch/
  README.md
  CODEX_SETUP.md
  PROGRAM.md
  BENCHMARK_SCHEMA.md
  SCORING.md
  .gitignore

  benchmarks/
    README.md
    cases/
      .gitkeep
    synthetic/
      .gitkeep
    fixtures/
      .gitkeep

  results/
    results.tsv
    best.json
    .gitkeep

  runs/
    .gitkeep

  scripts/
    __init__.py
    build_fixtures_from_quality_run.py
    compare_runs.py
    eval_translation_routing.py
    generate_synthetic_cases.py
    io_adapters.py
    score.py
    summarize_results.py
    validate_fixtures.py

  templates/
    case.example.json
    candidate_outputs.example.json
    references.example.json
    result_header.tsv
```

Put this setup spec itself inside the loop folder as:

```text
translation_routing_autoresearch/CODEX_SETUP.md
```

No setup files should be placed outside `translation_routing_autoresearch/`. The existing root-level `codex_translation_routing_autoresearch_setup.md` is only the handoff instruction file; the setup agent should copy its contents into `translation_routing_autoresearch/CODEX_SETUP.md` and should not create any other root-level setup files.

---

## 6. File responsibilities

### `translation_routing_autoresearch/README.md`

Explain the loop in human terms:

- It is isolated from the main project.
- It benchmarks only translation routing, validation, and source-risk classification.
- It uses frozen OCR text and frozen candidate translator outputs.
- It does not call live OCR, translation, Qwen, CAT, OPUS, rendering, or detection.
- It imports the main project as a library.
- It writes append-only results.
- It is intended to help future autoresearch agents decide which small source-code changes to keep or revert.

Include quick commands:

```bash
python translation_routing_autoresearch/scripts/generate_synthetic_cases.py \
  --output translation_routing_autoresearch/benchmarks/synthetic

python translation_routing_autoresearch/scripts/validate_fixtures.py \
  --benchmark translation_routing_autoresearch/benchmarks/synthetic

python translation_routing_autoresearch/scripts/eval_translation_routing.py \
  --benchmark translation_routing_autoresearch/benchmarks/synthetic \
  --output translation_routing_autoresearch/runs/baseline \
  --results translation_routing_autoresearch/results/results.tsv \
  --run-id baseline

python translation_routing_autoresearch/scripts/summarize_results.py \
  --results translation_routing_autoresearch/results/results.tsv
```

On Windows in this project, prefer the repo virtualenv when present:

```powershell
.\.venv\Scripts\python.exe translation_routing_autoresearch\scripts\generate_synthetic_cases.py `
  --output translation_routing_autoresearch\benchmarks\synthetic
```

### `translation_routing_autoresearch/CODEX_SETUP.md`

Copy this full setup spec into that file.

### `translation_routing_autoresearch/PROGRAM.md`

This is the agent-facing autoresearch instruction file. It should tell a future coding agent how to run experiments.

It must include:

- Objective: minimize `translation_routing_score`.
- Scope: routing, source-risk classification, validation, and safe fallback policy only.
- Frozen benchmark rule: do not modify benchmark cases or scoring during optimization.
- Required setup command.
- Required unit-test command.
- Required benchmark command.
- Editable files by phase.
- Forbidden edits.
- Keep/revert rules.
- Result-log requirements.

Use this objective statement:

```text
Minimize unsafe accepted translations while preserving coverage and minimizing unnecessary expensive fallback/review.
```

Use this phase-1 editable surface:

```text
manga_local_translator/translation_rules.py
manga_local_translator/translation_evidence.py
manga_local_translator/qwen_validation.py
manga_local_translator/text_filter.py
```

Use this phase-2 editable surface only after the phase-1 loop is stable:

```text
manga_local_translator/translate.py
manga_local_translator/hf_translators.py
manga_local_translator/qwen_translator.py
```

Forbidden edits during optimization:

```text
translation_routing_autoresearch/benchmarks/**
translation_routing_autoresearch/scripts/eval_translation_routing.py
translation_routing_autoresearch/scripts/score.py
translation_routing_autoresearch/templates/**
translation_routing_autoresearch/results/results.tsv
.translation caches or generated model outputs
.testing/tests/**
rendering/layout code unless explicitly requested later
OCR/detection code unless explicitly requested later
```

Required commands before accepting an experiment:

```powershell
.\.testing\run_tests.ps1

.\.venv\Scripts\python.exe translation_routing_autoresearch\scripts\validate_fixtures.py `
  --benchmark translation_routing_autoresearch\benchmarks\synthetic

.\.venv\Scripts\python.exe translation_routing_autoresearch\scripts\eval_translation_routing.py `
  --benchmark translation_routing_autoresearch\benchmarks\synthetic `
  --output translation_routing_autoresearch\runs\current `
  --results translation_routing_autoresearch\results\results.tsv `
  --run-id current `
  --tests-ok
```

Keep a change only if:

```text
1. unit tests pass,
2. fixture validation passes,
3. translation_routing_score improves versus current best,
4. unsafe_accepted_count does not increase,
5. prompt_chatter_accepted_count remains 0,
6. source_text_mutation_count remains 0,
7. line_id/source mapping errors remain 0,
8. accepted untranslated Japanese does not increase,
9. required term drops do not increase,
10. the improvement did not come from editing benchmark/scoring/logging files.
```

Otherwise revert the change and append a failed result row.

### `translation_routing_autoresearch/BENCHMARK_SCHEMA.md`

Define the fixture schema exactly. See section 8 below.

### `translation_routing_autoresearch/SCORING.md`

Define the scoring formula exactly. See section 10 below.

### `translation_routing_autoresearch/.gitignore`

Keep generated run noise out of commits, but keep schemas, scripts, benchmark definitions, and result logs visible.

Recommended content:

```gitignore
runs/*
!runs/.gitkeep

results/*.tmp
results/*.jsonl
results/*.log
!results/.gitkeep
!results/results.tsv
!results/best.json

__pycache__/
*.pyc
.DS_Store
```

### `translation_routing_autoresearch/results/results.tsv`

Create this file with the header from `templates/result_header.tsv`.

The log is append-only. Do not delete historical rows. Do not silently rewrite previous rows.

### `translation_routing_autoresearch/results/best.json`

Create a small JSON file tracking the current best run:

```json
{
  "schema_version": 1,
  "best_run_id": null,
  "best_score": null,
  "best_commit": null,
  "updated_at": null,
  "notes": "Updated by eval_translation_routing.py only when a run improves the score and passes hard guards."
}
```

---

## 7. Benchmarking concept

The benchmark should evaluate **routing decisions over frozen translation candidates**.

For each case, the evaluator loads:

```text
case metadata
source OCR text
neighbor text/context
OCR-risk labels
frozen candidate outputs
reference/acceptability rules
```

Then it asks the policy under test to choose:

```text
route
selected_output
needs_review
risk_flags
rejection_reason(s)
repair_reason(s)
```

The evaluator then scores that decision. It should not call any live model.

A policy can be evaluated in two modes:

```text
baseline mode:
  Use simple built-in routing in the harness plus imported project validators.

project mode:
  Import current project functions through io_adapters.py and evaluate the real routing/validation behavior where possible.
```

If the current project does not expose a single routing function, `io_adapters.py` should assemble one from available helpers. Keep this adapter thin and explicit.

---

## 8. Benchmark fixture schema

Each benchmark set is a folder containing three JSONL files:

```text
translation_routing_autoresearch/benchmarks/<set_name>/
  cases.jsonl
  candidate_outputs.jsonl
  references.jsonl
  README.md
```

For synthetic setup, create:

```text
translation_routing_autoresearch/benchmarks/synthetic/
  cases.jsonl
  candidate_outputs.jsonl
  references.jsonl
  README.md
```

For real cases built from quality/debug output, create:

```text
translation_routing_autoresearch/benchmarks/cases/<dataset_name>/
  cases.jsonl
  candidate_outputs.jsonl
  references.jsonl
  README.md
```

### `cases.jsonl`

One JSON object per OCR line/block:

```json
{
  "schema_version": 1,
  "case_id": "synthetic_short_fragment_001",
  "page_id": "synthetic_page_001",
  "line_id": "line_001",
  "block_id": "block_001",
  "source_text": "．．．そんな",
  "normalized_source_text": "…そんな",
  "neighbor_source_before": ["まさか"],
  "neighbor_source_after": ["ありえない"],
  "ocr_confidence": 42.0,
  "box": [120, 340, 190, 380],
  "source_type_label": "short_fragment",
  "risk_labels": ["short_fragment", "ellipsis_fragment", "ocr_risk"],
  "metadata": {
    "is_dialogue": true,
    "is_sfx": false,
    "is_credit_or_metadata": false,
    "orientation": "vertical",
    "notes": "Short fragment likely unsafe for freeform CAT."
  }
}
```

All fixture files containing Japanese text must be read and written as UTF-8. In Python, use `encoding="utf-8"` and `json.dumps(..., ensure_ascii=False)` so benchmark fixtures keep real Japanese text rather than console mojibake.

Required fields:

```text
schema_version
case_id
page_id
line_id
source_text
risk_labels
source_type_label
```

Optional but recommended fields:

```text
block_id
normalized_source_text
neighbor_source_before
neighbor_source_after
ocr_confidence
box
metadata
```

### `candidate_outputs.jsonl`

One JSON object per `case_id`, containing frozen candidate outputs from each possible route:

```json
{
  "schema_version": 1,
  "case_id": "synthetic_short_fragment_001",
  "candidates": {
    "phrasebook": {
      "available": true,
      "text": "...No way.",
      "source": "translate_known_phrase",
      "latency_proxy": 1
    },
    "opus": {
      "available": true,
      "text": "... such a thing",
      "source": "frozen_opus_output",
      "latency_proxy": 3
    },
    "cat_primary": {
      "available": true,
      "text": "Please provide the text you would like translated.",
      "source": "frozen_cat_output",
      "latency_proxy": 8
    },
    "cat_retry": {
      "available": true,
      "text": "No way...",
      "source": "frozen_cat_retry_output",
      "latency_proxy": 12
    },
    "qwen_critic": {
      "available": true,
      "text": "...No way.",
      "source": "frozen_qwen_critic_or_repair_output",
      "latency_proxy": 20
    },
    "q8_repair": {
      "available": false,
      "text": null,
      "source": null,
      "latency_proxy": 35
    },
    "human_review": {
      "available": true,
      "text": null,
      "source": "review_flag",
      "latency_proxy": 5
    },
    "skip": {
      "available": true,
      "text": "",
      "source": "skip_noise_or_metadata",
      "latency_proxy": 0
    }
  }
}
```

Candidate route names must be one of:

```text
skip
phrasebook
opus
cat_primary
cat_retry
qwen_critic
q8_repair
human_review
```

Do not add live model calls to fill these fields during evaluation. Fixture-building scripts may extract frozen outputs from prior runs, but the evaluator must never invoke a model.

### `references.jsonl`

One JSON object per `case_id`, defining acceptable behavior and scoring hints:

```json
{
  "schema_version": 1,
  "case_id": "synthetic_short_fragment_001",
  "acceptable_routes": ["phrasebook", "cat_retry", "qwen_critic", "human_review"],
  "preferred_routes": ["phrasebook", "cat_retry"],
  "forbidden_routes": ["cat_primary"],
  "acceptable_outputs": ["...No way.", "No way...", "That can't be..."],
  "forbidden_patterns": [
    "please provide",
    "I cannot",
    "as an AI",
    "I don't see",
    "translation:" 
  ],
  "must_preserve_terms": [],
  "allowed_japanese_output": false,
  "should_skip": false,
  "should_review": false,
  "unsafe_acceptance_labels": ["prompt_chatter", "hallucinated_dialogue"],
  "notes": "CAT primary chatter must not be accepted."
}
```

Required fields:

```text
schema_version
case_id
acceptable_routes
forbidden_patterns
must_preserve_terms
allowed_japanese_output
should_skip
should_review
```

Optional fields:

```text
preferred_routes
forbidden_routes
acceptable_outputs
unsafe_acceptance_labels
notes
```

---

## 9. Synthetic benchmark cases to generate

`generate_synthetic_cases.py` must create at least 40 cases covering these categories:

```text
short fragments
ellipsis fragments
SFX / onomatopoeia
katakana name fragments
honorifics
noisy credit text
metadata / page numbers
OCR-corrupted kana
OCR-corrupted playful phrases
untranslated Japanese leakage
prompt chatter from model output
assistant refusal text
overlong hallucinated rescue
wrong-line mapping risk
invented names
invented relationships
gender-inference risk
number preservation
code-like bracket preservation such as 〈PII2〉
quote/marker salvage
broad unsafe salvage
empty or punctuation-only OCR
mixed Japanese/Latin noise
ruby/furigana-like fragments
speaker-neutral Japanese group address
```

The synthetic cases should be small, deterministic, and cheap. They are not intended to prove final translation quality; they are intended to catch routing and validation regressions.

Examples of bad outputs that should be rejected when accepted by a route:

```text
"Please provide the text you would like translated."
"I cannot translate the image without more context."
"As an AI language model..."
"It's a beautiful day today." for noisy OCR
invented proper names not present in source/glossary
English that drops required bracket/code terms
English that changes numbers
Japanese output when Japanese leakage is not allowed
```

---

## 10. Scoring

Create `translation_routing_autoresearch/SCORING.md` and `translation_routing_autoresearch/scripts/score.py` with this formula.

Primary metric:

```text
translation_routing_score
```

Lower is better.

Formula:

```text
translation_routing_score =
  10000 * unsafe_accepted_rate
+  9000  * line_mapping_error_rate
+  8500  * source_text_mutation_rate
+  8000  * hallucinated_dialogue_rate
+  7000  * prompt_chatter_accepted_rate
+  6500  * japanese_leakage_accepted_rate
+  5500  * required_term_drop_rate
+  4500  * invented_identity_rate
+  3500  * wrong_route_hard_rate
+  2200  * bad_english_accepted_rate
+  1400  * false_reject_rate
+  1000  * unnecessary_expensive_fallback_rate
+   600  * unnecessary_review_rate
+    20  * mean_latency_proxy
```

Definitions:

```text
unsafe_accepted_rate:
  accepted output violates any hard safety/reference rule.

line_mapping_error_rate:
  selected output is associated with the wrong line_id, missing line_id, duplicate line_id, or wrong case_id.

source_text_mutation_rate:
  policy/model output mutates source_text or normalized_source_text when preserving original OCR identity is required.

hallucinated_dialogue_rate:
  selected output adds dialogue/facts not supported by source/reference.

prompt_chatter_accepted_rate:
  selected output contains assistant behavior, refusals, instruction chatter, or prompt residue.

japanese_leakage_accepted_rate:
  selected output contains Japanese text where `allowed_japanese_output` is false.

required_term_drop_rate:
  selected output drops a required number, code-like bracket term, name marker, or glossary term listed in `must_preserve_terms`.

invented_identity_rate:
  selected output invents a speaker name, relationship, gender, or identity not grounded by the source/reference.

wrong_route_hard_rate:
  selected route is listed in `forbidden_routes`.

bad_english_accepted_rate:
  selected output is empty, malformed, stage-direction-only when dialogue is expected, or obvious broken English.

false_reject_rate:
  case was safe and had an acceptable cheap output, but the policy skipped or sent it to human review unnecessarily.

unnecessary_expensive_fallback_rate:
  case had a preferred cheap route available, but the policy selected Qwen/Q8/human_review without a hard reason.

unnecessary_review_rate:
  case was marked human_review even though a preferred acceptable route was available.

mean_latency_proxy:
  mean of selected candidate `latency_proxy` values.
```

Hard failure conditions:

```text
unit tests fail
fixture validation fails
evaluator crashes
results row is not appended
benchmark/scoring files were edited during an optimization run
any selected output has wrong case_id/line_id
source_text mutation count > 0
prompt_chatter_accepted_count > 0
invalid JSON in per-case report
NaN or missing score
```

Tie-breakers, in order:

```text
1. lower unsafe_accepted_count
2. lower prompt_chatter_accepted_count
3. lower japanese_leakage_accepted_count
4. lower required_term_drop_count
5. lower false_reject_count
6. lower qwen_or_q8_usage_count
7. lower human_review_count
8. lower mean_latency_proxy
9. smaller project-code diff
```

---

## 11. Result log schema

Create:

```text
translation_routing_autoresearch/templates/result_header.tsv
translation_routing_autoresearch/results/results.tsv
```

Both should use exactly this header:

```text
run_id	timestamp	commit	parent_commit	experiment_name	changed_files	tests_ok	fixture_validation_ok	benchmark_set	translation_routing_score	baseline_score	delta_score	cases_total	unsafe_accepted_count	line_mapping_error_count	source_text_mutation_count	hallucinated_dialogue_count	prompt_chatter_accepted_count	japanese_leakage_accepted_count	required_term_drop_count	invented_identity_count	wrong_route_hard_count	bad_english_accepted_count	false_reject_count	unnecessary_expensive_fallback_count	unnecessary_review_count	skip_count	phrasebook_count	opus_count	cat_primary_count	cat_retry_count	qwen_critic_count	q8_repair_count	human_review_count	mean_latency_proxy	kept	notes
```

Rules:

```text
Append exactly one row per benchmark run.
Never delete previous rows.
Never silently rewrite previous rows.
Use ISO-8601 timestamps.
Use current git commit hash if available; otherwise write UNKNOWN.
Use semicolon-separated paths in changed_files.
Use TRUE/FALSE for booleans.
Use numeric zero instead of blanks for counts.
```

---

## 12. Run output schema

Each run writes to:

```text
translation_routing_autoresearch/runs/<run_id>/
```

Required files:

```text
summary.json
per_case_metrics.jsonl
selected_outputs.jsonl
failures.jsonl
config.json
```

### `summary.json`

Example:

```json
{
  "schema_version": 1,
  "run_id": "baseline",
  "benchmark_set": "synthetic",
  "cases_total": 40,
  "translation_routing_score": 1234.56,
  "hard_failure": false,
  "metrics": {
    "unsafe_accepted_count": 0,
    "prompt_chatter_accepted_count": 0,
    "false_reject_count": 3,
    "qwen_critic_count": 4,
    "human_review_count": 2,
    "mean_latency_proxy": 4.2
  }
}
```

### `per_case_metrics.jsonl`

Each line should include:

```json
{
  "case_id": "synthetic_short_fragment_001",
  "line_id": "line_001",
  "selected_route": "phrasebook",
  "selected_output": "...No way.",
  "accepted": true,
  "needs_review": false,
  "risk_flags": ["short_fragment"],
  "violations": [],
  "latency_proxy": 1,
  "case_score": 0.0
}
```

### `selected_outputs.jsonl`

Each line should include the final decision for manual inspection.

### `failures.jsonl`

Only cases with violations should be written here.

---

## 13. Script responsibilities

### `scripts/validate_fixtures.py`

Validate all fixture files before evaluation.

Checks:

```text
cases.jsonl exists
candidate_outputs.jsonl exists
references.jsonl exists
case_id sets match across all files
required fields exist
candidate route names are valid
latency_proxy values are numeric
references contain required fields
JSON lines parse cleanly
no duplicate case_id
```

Exit nonzero on validation failure.

### `scripts/generate_synthetic_cases.py`

Generate deterministic synthetic benchmark fixtures.

Requirements:

```text
no model calls
no network calls
stable output order
at least 40 cases
covers all categories in section 9
writes README.md explaining synthetic nature
```

### `scripts/build_fixtures_from_quality_run.py`

Optional utility for converting existing debug/quality-run artifacts into frozen benchmark cases.

Requirements:

```text
read-only access to source quality-run folder
no model calls
copy only compact fixture data into translation_routing_autoresearch/benchmarks/cases/<dataset_name>/
do not depend on the original quality-run folder at evaluation time
preserve source line_id and source_text exactly
```

### `scripts/io_adapters.py`

Provide one narrow interface used by the evaluator:

```python
def route_translation_case(case: dict, candidates: dict) -> dict:
    """Return selected route/output/review/risk info for one frozen case."""
```

The return value must be shaped like:

```json
{
  "case_id": "synthetic_short_fragment_001",
  "line_id": "line_001",
  "selected_route": "phrasebook",
  "selected_output": "...No way.",
  "accepted": true,
  "needs_review": false,
  "risk_flags": ["short_fragment"],
  "rejection_reasons": [],
  "repair_reasons": [],
  "latency_proxy": 1
}
```

Adapter rules:

```text
Prefer importing existing project helpers.
Do not require a live model.
Do not call network services.
Do not call Ollama, llama.cpp, Qwen, CAT, OPUS, MADLAD, Argos, OCR, or rendering.
If project helpers are unavailable or signatures changed, fall back to a simple deterministic baseline and record adapter_notes in config.json.
```

### `scripts/score.py`

Pure scoring functions only.

Requirements:

```text
no project imports
no model calls
no file writes except through caller
calculate all per-case violations
calculate summary score
apply hard failure flags
```

### `scripts/eval_translation_routing.py`

Main benchmark runner.

Responsibilities:

```text
parse arguments
validate fixtures
load cases/candidates/references
call io_adapters.route_translation_case for each case
score each decision
write run outputs
append one results.tsv row
update best.json only if score improves and hard guards pass
exit nonzero on hard failure unless --allow-hard-failure is passed
```

Recommended CLI:

```bash
python translation_routing_autoresearch/scripts/eval_translation_routing.py \
  --benchmark translation_routing_autoresearch/benchmarks/synthetic \
  --output translation_routing_autoresearch/runs/baseline \
  --results translation_routing_autoresearch/results/results.tsv \
  --run-id baseline \
  --experiment-name baseline
```

Required arguments:

```text
--benchmark
--output
--results
--run-id
```

Optional arguments:

```text
--experiment-name
--baseline-score
--allow-hard-failure
--no-update-best
```

### `scripts/compare_runs.py`

Compare two run folders:

```bash
python translation_routing_autoresearch/scripts/compare_runs.py \
  --before translation_routing_autoresearch/runs/baseline \
  --after translation_routing_autoresearch/runs/current
```

Output:

```text
score delta
metric deltas
cases improved
cases regressed
new hard failures
new unsafe acceptances
new false rejects
route distribution changes
```

### `scripts/summarize_results.py`

Read `results.tsv` and print:

```text
best kept run
latest run
score trend
top regressions
route distribution
unsafe acceptance trend
review/fallback trend
```

No dependency on pandas is required.

---

## 14. Baseline behavior

The first baseline can be simple and deterministic. A reasonable baseline policy:

```text
1. If source is empty, punctuation-only, page number, or known metadata: route skip.
2. If translate_known_phrase(source_text) returns a known phrase: route phrasebook.
3. If source/risk labels include credit/name/noisy_ocr: route human_review unless a safe phrasebook candidate exists.
4. If CAT primary output contains chatter/refusal/Japanese leakage: reject CAT primary.
5. If CAT retry is available and clean: route cat_retry.
6. If OPUS is clean and the case is low risk: route opus.
7. If Qwen critic or Q8 repair candidate is available for high-risk cases: route qwen_critic or q8_repair.
8. Otherwise route human_review.
```

This baseline should not be tuned to get a perfect synthetic score. It should leave room for future autoresearch improvements.

---

## 15. First experiment queue for future autoresearch

Add this queue to `PROGRAM.md`:

```text
1. Source-risk classifier improvement
   Better distinguish clean dialogue, short fragments, SFX, credit/name-only text, noisy OCR, bracket/code terms, and high hallucination-risk fragments.

2. CAT eligibility policy
   Prevent raw CAT from accepting risky short/noisy lines while preserving CAT use for clean dialogue.

3. Phrasebook-before-model policy
   Route common SFX, ellipsis, and very short manga fragments through deterministic phrasebook rules before expensive/model routes.

4. Conservative CAT salvage
   Allow only marker-backed or evidence-backed salvage. Reject broad quote salvage that turns chatter into plausible hallucinations.

5. Qwen/Q8 trigger policy
   Trigger Qwen/Q8 only when local evidence says the line is risky enough or a required preservation check failed.

6. Required-term preservation
   Improve detection of dropped numbers, code-like bracket terms, honorifics, and glossary terms.

7. Review threshold tuning
   Reduce false human-review routing without allowing unsafe acceptances.
```

---

## 16. Guardrails against metric gaming

Do not allow the optimization agent to improve the score by:

```text
editing benchmark cases
editing references
editing candidate outputs
editing score weights
hiding violations
changing result-log schema
routing everything to skip
routing everything to human_review
routing everything to Qwen/Q8
accepting prompt chatter because it is fluent English
allowing source_text mutation
removing Japanese-leakage checks
removing required-term preservation checks
```

The evaluator should include explicit route-distribution penalties so an agent cannot score well by routing every case to the most conservative or most expensive route.

---

## 17. Required self-check after setup

After creating all files, run:

```powershell
.\.venv\Scripts\python.exe translation_routing_autoresearch\scripts\generate_synthetic_cases.py `
  --output translation_routing_autoresearch\benchmarks\synthetic

.\.venv\Scripts\python.exe translation_routing_autoresearch\scripts\validate_fixtures.py `
  --benchmark translation_routing_autoresearch\benchmarks\synthetic

.\.venv\Scripts\python.exe translation_routing_autoresearch\scripts\eval_translation_routing.py `
  --benchmark translation_routing_autoresearch\benchmarks\synthetic `
  --output translation_routing_autoresearch\runs\baseline `
  --results translation_routing_autoresearch\results\results.tsv `
  --run-id baseline `
  --experiment-name setup_baseline `
  --tests-ok

.\.venv\Scripts\python.exe translation_routing_autoresearch\scripts\summarize_results.py `
  --results translation_routing_autoresearch\results\results.tsv
```

The setup is not complete until:

```text
synthetic fixtures exist
fixture validation passes
baseline run exists
results.tsv has exactly one baseline row after first setup run
summary.json exists
per_case_metrics.jsonl exists
best.json is valid JSON
all files are under translation_routing_autoresearch/
```

---

## 18. Final setup checklist

Before stopping, confirm:

```text
[ ] No files were created outside translation_routing_autoresearch/.
[ ] No files in manga_local_translator/ were edited during setup.
[ ] No files in .testing/ were edited during setup.
[ ] No model calls are made by the evaluator.
[ ] No network calls are made by the evaluator.
[ ] Synthetic benchmark has at least 40 cases.
[ ] Fixture schemas are documented.
[ ] Scoring formula is documented.
[ ] results.tsv has the required header.
[ ] best.json exists and is valid JSON.
[ ] PROGRAM.md contains editable scope and keep/revert rules.
[ ] README.md contains commands for generating, validating, evaluating, and summarizing.
[ ] eval_translation_routing.py appends exactly one row per run.
[ ] Benchmark and scoring files are clearly marked read-only for future optimization agents.
```

---

## 19. Summary for Codex

Set up a root-level folder named:

```text
translation_routing_autoresearch/
```

Inside it, build a self-contained autoresearch harness that benchmarks routing decisions over frozen OCR text and frozen translation candidates. The harness should score unsafe accepted translations, prompt chatter, Japanese leakage, required-term drops, invented identity/facts, false rejects, unnecessary expensive fallback, and unnecessary human review.

Keep the loop completely separate from the main project. During setup, create only `translation_routing_autoresearch/**`. Future experiments may edit a narrow set of translation/routing files, but the benchmark, scoring, results, and run logs must stay isolated and read-only.
