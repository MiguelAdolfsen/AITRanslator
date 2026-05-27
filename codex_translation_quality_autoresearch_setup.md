# Codex Implementation Brief: Translation Quality Autoresearch Harness for AITRanslator

## Purpose

Build a new, isolated autoresearch harness that improves translation quality in the AITRanslator pipeline.

This harness must evaluate and optimize only this stage:

```text
validated OCR source blocks + context -> translator agents -> final English translation
```

It must not evaluate or optimize:

```text
detection
OCR
source extraction
block grouping
rendering
erase / inpainting
GUI behavior
```

Those stages already have or can have their own isolated autoresearch loops. This loop is for translation-agent quality only.

The main implementation objective is to create a reproducible benchmark and scoring harness that can compare translator-agent outputs, record traces, score failures, and guide future prompt/rule/reranking optimization.

---

## Absolute isolation requirement

Create all harness files in one new root-level folder:

```text
translation_quality_autoresearch/
```

Do not place harness files inside:

```text
manga_local_translator/
.testing/
quality-runs/
render_autoresearch/
source_extraction_autoresearch/
cat_response_autoresearch/
autoresearch_docs/
```

The harness may import `manga_local_translator` modules and may read existing debug outputs, cache files, and quality runs, but its own benchmark, scripts, scores, traces, reports, and docs must stay inside:

```text
translation_quality_autoresearch/
```

The harness must be separable. Deleting `translation_quality_autoresearch/` should leave the main project behavior unchanged.

---

## Design principles

1. **Frozen benchmark first.** Translation experiments must run on fixed source cases. Do not let detection/OCR/rendering noise affect translation scoring.
2. **Trace everything.** Record all candidates, prompts, agent names, repair decisions, judge outputs, scores, and final selection reasons.
3. **Replay mode before live mode.** The first version must evaluate frozen candidate outputs without calling CAT/Qwen/OPUS. Live agent execution comes after replay evaluation works.
4. **Hard checks before model judges.** Deterministic checks must catch assistant chatter, Japanese leakage, line mapping breakage, glossary violations, invalid JSON, and empty outputs before any LLM judge or COMET score is used.
5. **No single metric is trusted alone.** Use a composite score combining deterministic checks, MQM-style error categories, optional MT metrics, pairwise comparison, cost, and latency.
6. **Lower score is better.** Every result row must expose one primary metric named `translation_quality_score`.
7. **No benchmark cheating.** Future autoresearch agents may edit translator prompts/rules/reranking code, but must not edit benchmark cases, scorer code, or result logs to improve scores.
8. **Local-first.** The harness should work without external APIs. Optional external or heavyweight scorers may be skipped cleanly when unavailable.
9. **No dependency pollution.** Do not modify the root `requirements.txt` for the initial harness. Optional dependencies belong in the harness folder.
10. **Small, inspectable outputs.** Every run should generate machine-readable JSON/JSONL/TSV and enough summaries to inspect why the score changed.

---

## Repository files Codex should inspect first

Before implementing, inspect these files and adapt imports/function calls to the actual codebase:

```text
README.md
pyproject.toml
requirements.txt
translation_glossary.example.json
manga_local_translator/cli.py
manga_local_translator/config.py
manga_local_translator/pipeline.py
manga_local_translator/quality_eval.py
manga_local_translator/hf_translators.py
manga_local_translator/qwen_translator.py
manga_local_translator/qwen_validation.py
manga_local_translator/qwen_vision.py
manga_local_translator/vision_service.py
manga_local_translator/grouping.py
manga_local_translator/debug_report.py
```

Also inspect the existing isolated autoresearch folders so the new folder follows the repo's style:

```text
render_autoresearch/
source_extraction_autoresearch/
cat_response_autoresearch/
```

Do not copy their scoring formulas blindly; only mirror useful conventions such as isolated scripts, `PROGRAM.md`, `README.md`, `runs/`, `results/`, and append-only TSV logs.

---

## Required folder tree

Create this exact root-level structure:

```text
translation_quality_autoresearch/
  README.md
  PROGRAM.md
  IMPLEMENTATION_NOTES.md

  benchmark/
    README.md
    cases.jsonl
    references.jsonl
    frozen_agent_outputs.jsonl
    glossary.json
    human_gold.jsonl
    adversarial_cases.jsonl
    page_context_cases.jsonl
    seed_cases.jsonl

  config/
    default_eval.json
    default_live_agents.json
    scoring_weights.json
    judge_rubric_mqm.json
    reranker_weights.json

  common/
    __init__.py
    schemas.py
    io_utils.py
    text_normalize.py
    hashing.py
    project_imports.py
    metrics.py
    result_log.py

  scorers/
    __init__.py
    deterministic_checks.py
    japanese_leakage.py
    glossary_checks.py
    manga_style_checks.py
    mqm_judge.py
    pairwise_judge.py
    comet_scorer.py
    score_formula.py

  scripts/
    build_cases_from_quality_runs.py
    make_seed_benchmark.py
    run_translation_agents.py
    eval_translation_quality.py
    rerank_candidates.py
    summarize_run.py
    compare_runs.py
    validate_benchmark.py

  optimizers/
    __init__.py
    prompt_search.py
    failure_reflection.py
    dspy_mipro_adapter.py
    gepa_adapter.py

  tests/
    __init__.py
    test_schemas.py
    test_deterministic_checks.py
    test_score_formula.py
    test_result_log.py
    test_reranker.py

  runs/
    .gitkeep

  results/
    translation_quality_results.tsv
    .gitkeep

  prompts/
    README.md
    candidate_translation_prompt.txt
    page_context_translation_prompt.txt
    critic_prompt.txt
    repair_prompt.txt
    mqm_judge_prompt.txt
    pairwise_judge_prompt.txt

  optional_requirements.txt
```

Notes:

- `benchmark/*.jsonl` should contain small seed examples initially, but the scripts must support larger real datasets later.
- `runs/` is for per-run artifacts.
- `results/translation_quality_results.tsv` is append-only.
- `optional_requirements.txt` may list optional packages such as COMET/DSPy, but the harness must run without them.
- `common/`, `scorers/`, `scripts/`, and `optimizers/` are harness modules, not main-project modules.

---

## Required README.md content

Create `translation_quality_autoresearch/README.md` explaining:

```text
What this harness does
What it does not do
How it stays separate from the main project
How to run replay evaluation
How to run live agent generation
How to inspect a run
How to append result logs
How to add benchmark cases
How to avoid benchmark leakage
How to run tests
```

The README must include these commands:

```powershell
# Validate benchmark schemas
.\.venv\Scripts\python.exe translation_quality_autoresearch\scripts\validate_benchmark.py `
  --benchmark translation_quality_autoresearch\benchmark

# Run unit tests for the harness
.\.venv\Scripts\python.exe -m unittest discover -s translation_quality_autoresearch\tests

# Run replay-mode evaluation with frozen outputs
.\.venv\Scripts\python.exe translation_quality_autoresearch\scripts\eval_translation_quality.py `
  --mode replay `
  --cases translation_quality_autoresearch\benchmark\cases.jsonl `
  --references translation_quality_autoresearch\benchmark\references.jsonl `
  --frozen-outputs translation_quality_autoresearch\benchmark\frozen_agent_outputs.jsonl `
  --glossary translation_quality_autoresearch\benchmark\glossary.json `
  --output translation_quality_autoresearch\runs\baseline_replay `
  --results translation_quality_autoresearch\results\translation_quality_results.tsv `
  --run-id baseline_replay `
  --overwrite-output

# Generate live agent candidates, if local models are available
.\.venv\Scripts\python.exe translation_quality_autoresearch\scripts\run_translation_agents.py `
  --cases translation_quality_autoresearch\benchmark\cases.jsonl `
  --agents opus,qwen_block,qwen_page,cat `
  --output translation_quality_autoresearch\runs\live_candidates\agent_traces.jsonl

# Evaluate a live trace
.\.venv\Scripts\python.exe translation_quality_autoresearch\scripts\eval_translation_quality.py `
  --mode live-trace `
  --cases translation_quality_autoresearch\benchmark\cases.jsonl `
  --references translation_quality_autoresearch\benchmark\references.jsonl `
  --traces translation_quality_autoresearch\runs\live_candidates\agent_traces.jsonl `
  --glossary translation_quality_autoresearch\benchmark\glossary.json `
  --output translation_quality_autoresearch\runs\live_eval `
  --results translation_quality_autoresearch\results\translation_quality_results.tsv `
  --run-id live_eval `
  --overwrite-output
```

Also include Unix-style variants:

```bash
python translation_quality_autoresearch/scripts/validate_benchmark.py \
  --benchmark translation_quality_autoresearch/benchmark

python -m unittest discover -s translation_quality_autoresearch/tests

python translation_quality_autoresearch/scripts/eval_translation_quality.py \
  --mode replay \
  --cases translation_quality_autoresearch/benchmark/cases.jsonl \
  --references translation_quality_autoresearch/benchmark/references.jsonl \
  --frozen-outputs translation_quality_autoresearch/benchmark/frozen_agent_outputs.jsonl \
  --glossary translation_quality_autoresearch/benchmark/glossary.json \
  --output translation_quality_autoresearch/runs/baseline_replay \
  --results translation_quality_autoresearch/results/translation_quality_results.tsv \
  --run-id baseline_replay \
  --overwrite-output
```

---

## Required PROGRAM.md content

Create `translation_quality_autoresearch/PROGRAM.md` as the instruction file for future autoresearch agents.

It must say:

```md
# Translation Quality Autoresearch Program

Goal: minimize `translation_quality_score` on the fixed benchmark.

This loop optimizes translation-agent quality only. It must not optimize detection, OCR, grouping, erase, rendering, GUI behavior, or benchmark scoring.

## Read first

Before every experiment, read:

- README.md
- translation_quality_autoresearch/README.md
- translation_quality_autoresearch/PROGRAM.md
- translation_quality_autoresearch/config/scoring_weights.json
- translation_quality_autoresearch/scorers/score_formula.py
- manga_local_translator/qwen_translator.py
- manga_local_translator/qwen_validation.py
- manga_local_translator/hf_translators.py
- manga_local_translator/quality_eval.py
- manga_local_translator/pipeline.py

## Benchmark isolation

Do not edit:

- translation_quality_autoresearch/benchmark/*
- translation_quality_autoresearch/scorers/*
- translation_quality_autoresearch/scripts/eval_translation_quality.py
- translation_quality_autoresearch/results/*
- translation_quality_autoresearch/tests/*

unless the task is explicitly to improve the harness itself.

## Allowed edits for translation-quality experiments

Primary phase:

- manga_local_translator/qwen_translator.py
- manga_local_translator/qwen_validation.py
- manga_local_translator/hf_translators.py
- translation_quality_autoresearch/prompts/*.txt
- translation_quality_autoresearch/config/reranker_weights.json

Secondary phase, only when explicitly enabled:

- manga_local_translator/pipeline.py
- manga_local_translator/config.py
- manga_local_translator/quality_eval.py

Forbidden for translation-quality experiments:

- detection/OCR code
- grouping code
- render code
- erase code
- benchmark labels
- scoring code
- result logs except append-only run rows

## Required checks

Run:

```bash
python -m unittest discover -s translation_quality_autoresearch/tests
python translation_quality_autoresearch/scripts/validate_benchmark.py --benchmark translation_quality_autoresearch/benchmark
python translation_quality_autoresearch/scripts/eval_translation_quality.py --mode replay --cases translation_quality_autoresearch/benchmark/cases.jsonl --references translation_quality_autoresearch/benchmark/references.jsonl --frozen-outputs translation_quality_autoresearch/benchmark/frozen_agent_outputs.jsonl --glossary translation_quality_autoresearch/benchmark/glossary.json --output translation_quality_autoresearch/runs/current --results translation_quality_autoresearch/results/translation_quality_results.tsv --run-id current --overwrite-output
```

Keep a change only if:

- tests pass
- benchmark validation passes
- eval completes
- `translation_quality_score` improves versus current best
- critical MQM error rate does not increase
- Japanese leakage does not increase
- assistant chatter does not appear
- line/source mapping remains stable
- glossary violations do not increase

Otherwise revert and log the failure.
```

---

## Data model

Implement schemas in:

```text
translation_quality_autoresearch/common/schemas.py
```

Use Python dataclasses or typed dictionaries. Keep them dependency-light. Do not require Pydantic unless it is already installed.

### TranslationCase

Each line in `benchmark/cases.jsonl` must match this shape:

```json
{
  "case_id": "seed_clean_dialogue_001",
  "page_id": "seed_page_001",
  "group_id": "g001",
  "source_text": "まったく…しょうがないな",
  "normalized_source": "まったく…しょうがないな",
  "source_hash": "sha256:...",
  "context_before": ["遅いぞ"],
  "context_after": ["早く行くよ"],
  "speaker_hint": "unknown",
  "ocr_risk": "clean",
  "source_type": "dialogue",
  "orientation": "vertical",
  "box": [120, 240, 180, 420],
  "page_order": 4,
  "glossary_terms": [],
  "must_preserve": [],
  "forbidden_patterns": ["I cannot", "please provide", "as an AI", "Translation:"],
  "style_notes": ["natural manga dialogue", "do not over-explain", "do not add speaker names"],
  "tags": ["clean_dialogue", "ellipsis", "casual"],
  "metadata": {
    "source_file": "seed",
    "difficulty": "easy"
  }
}
```

Required fields:

```text
case_id
source_text
source_hash
source_type
ocr_risk
context_before
context_after
forbidden_patterns
must_preserve
tags
```

Optional fields:

```text
page_id
group_id
normalized_source
speaker_hint
orientation
box
page_order
glossary_terms
style_notes
metadata
```

### ReferenceRecord

Each line in `benchmark/references.jsonl` must match this shape:

```json
{
  "case_id": "seed_clean_dialogue_001",
  "reference_translations": [
    "Honestly... you’re hopeless.",
    "Seriously... what am I going to do with you?"
  ],
  "acceptable_meaning_notes": [
    "Speaker is exasperated but affectionate.",
    "Do not add a specific name or gender."
  ],
  "known_bad_translations": [
    "As an AI, I cannot translate this.",
    "The teacher named John is angry."
  ],
  "required_meaning_units": ["exasperation", "can't be helped"],
  "forbidden_additions": ["teacher", "John", "school"],
  "human_priority": "normal"
}
```

### FrozenAgentOutput

Each line in `benchmark/frozen_agent_outputs.jsonl` should contain all frozen candidates for one case:

```json
{
  "case_id": "seed_clean_dialogue_001",
  "source_hash": "sha256:...",
  "candidates": [
    {
      "candidate_id": "seed_clean_dialogue_001__opus",
      "agent": "opus",
      "text": "I can't help it...",
      "raw_output": "I can't help it...",
      "prompt_name": null,
      "model_name": "opus-mt-ja-en",
      "latency_ms": 0,
      "cost_proxy": 1.0,
      "metadata": {}
    },
    {
      "candidate_id": "seed_clean_dialogue_001__qwen_page",
      "agent": "qwen_page",
      "text": "Honestly... what am I going to do with you?",
      "raw_output": "Honestly... what am I going to do with you?",
      "prompt_name": "page_context_translation_prompt",
      "model_name": "local-qwen",
      "latency_ms": 0,
      "cost_proxy": 4.0,
      "metadata": {}
    }
  ]
}
```

### AgentTraceRecord

Live agent runs must write JSONL records shaped like this:

```json
{
  "case_id": "seed_clean_dialogue_001",
  "source_hash": "sha256:...",
  "run_id": "live_candidates_001",
  "timestamp": "2026-05-27T12:00:00+02:00",
  "candidates": [
    {
      "candidate_id": "seed_clean_dialogue_001__qwen_block__001",
      "agent": "qwen_block",
      "text": "Honestly... you're hopeless.",
      "raw_output": "{...raw model text...}",
      "prompt_name": "candidate_translation_prompt",
      "prompt_hash": "sha256:...",
      "model_name": "Qwen3-8B-Q4_K_M.gguf",
      "latency_ms": 3480,
      "cost_proxy": 4.0,
      "accepted_by_agent": true,
      "agent_warnings": [],
      "metadata": {
        "temperature": 0.2,
        "max_tokens": 256
      }
    }
  ]
}
```

### CandidateScore

`eval_translation_quality.py` should write per-candidate scores to:

```text
translation_quality_autoresearch/runs/<run_id>/candidate_scores.jsonl
```

Shape:

```json
{
  "case_id": "seed_clean_dialogue_001",
  "candidate_id": "seed_clean_dialogue_001__qwen_page",
  "agent": "qwen_page",
  "text": "Honestly... what am I going to do with you?",
  "deterministic": {
    "hard_fail": false,
    "assistant_chatter": false,
    "japanese_leakage": false,
    "empty_output": false,
    "source_copied": false,
    "forbidden_pattern": false,
    "glossary_violation": false,
    "line_mapping_error": false,
    "oververbose": false,
    "warnings": []
  },
  "mqm": {
    "enabled": false,
    "critical_errors": 0,
    "major_errors": 0,
    "minor_errors": 0,
    "errors": []
  },
  "mt_metrics": {
    "comet": null,
    "cometkiwi": null,
    "xcomet": null
  },
  "style": {
    "length_ratio": 1.15,
    "punctuation_ok": true,
    "dialogue_naturalness_proxy": 0.8
  },
  "candidate_quality_score": 123.4,
  "hard_reject": false
}
```

### FinalCaseDecision

`eval_translation_quality.py` should write final selected candidate decisions to:

```text
translation_quality_autoresearch/runs/<run_id>/case_decisions.jsonl
```

Shape:

```json
{
  "case_id": "seed_clean_dialogue_001",
  "source_hash": "sha256:...",
  "selected_candidate_id": "seed_clean_dialogue_001__qwen_page",
  "selected_agent": "qwen_page",
  "selected_text": "Honestly... what am I going to do with you?",
  "selection_reason": "lowest candidate score; no hard warnings; better style proxy",
  "rejected_candidates": [
    {
      "candidate_id": "seed_clean_dialogue_001__opus",
      "reason": "higher omission/style penalty"
    }
  ],
  "case_score": 123.4,
  "hard_failures": [],
  "warnings": []
}
```

---

## Benchmark subsets

The benchmark folder should support multiple subsets. Implement loaders so `--cases` can accept either one JSONL file or a comma-separated list of JSONL files.

Initial files:

```text
benchmark/cases.jsonl
benchmark/adversarial_cases.jsonl
benchmark/page_context_cases.jsonl
benchmark/seed_cases.jsonl
```

Recommended future subsets:

```text
clean_dialogue.jsonl
short_fragments.jsonl
sfx_and_reactions.jsonl
ambiguous_subjects.jsonl
names_and_honorifics.jsonl
glossary_sensitive.jsonl
ocr_noisy_but_salvageable.jsonl
page_context_required.jsonl
hallucination_traps.jsonl
vision_context_cases.jsonl
```

Case tags must be used for per-subset reporting. For example:

```json
"tags": ["hallucination_trap", "ambiguous_speaker", "short"]
```

The evaluation summary must report score slices by tag.

---

## Seed benchmark content

Add a small non-copyright seed benchmark that tests the scorer and harness. These should be synthetic Japanese-like or common short phrases, not copied manga text.

Add at least 12 cases:

1. Clean casual dialogue.
2. Ellipsis-heavy short fragment.
3. Sound effect / reaction.
4. Ambiguous subject with no gender.
5. Name/honorific preservation.
6. Glossary-sensitive term.
7. OCR-risk noisy source.
8. Page-context-required response.
9. Hallucination trap with no speaker name.
10. Forbidden assistant-chatter candidate.
11. Japanese leakage candidate.
12. Oververbose explanation candidate.

Each seed case must have at least three frozen candidates:

```text
good candidate
minor-error candidate
hard-failure candidate
```

This lets unit tests verify that scoring and reranking choose the correct candidate before live models are used.

---

## Text normalization

Implement in:

```text
translation_quality_autoresearch/common/text_normalize.py
```

Functions:

```python
def normalize_source_japanese(text: str) -> str: ...
def normalize_target_english(text: str) -> str: ...
def stable_text_hash(text: str) -> str: ...
def contains_japanese(text: str) -> bool: ...
def japanese_char_ratio(text: str) -> float: ...
def looks_like_assistant_chatter(text: str) -> bool: ...
def strip_translation_prefixes(text: str) -> str: ...
```

Rules:

- Normalize Unicode with NFKC where appropriate.
- Preserve Japanese punctuation for source hashing.
- Normalize whitespace.
- Treat hiragana, katakana, and CJK unified ideographs as Japanese.
- Do not remove source content in a way that changes hashes invisibly.
- Store `source_hash` as `sha256:<hex>` over normalized source text.

Hard-pattern examples for assistant chatter:

```text
as an AI
I cannot translate
I can't translate
please provide
I need more context
translation:
here is the translation
cannot determine from the image
I'm sorry
```

The deterministic checker should be case-insensitive for English boilerplate.

---

## Deterministic checks

Implement in:

```text
translation_quality_autoresearch/scorers/deterministic_checks.py
```

Function:

```python
def run_deterministic_checks(case: TranslationCase, candidate: Candidate, glossary: dict | None) -> DeterministicResult:
    ...
```

Checks:

```text
empty_output
assistant_chatter
japanese_leakage
source_copied
forbidden_pattern
invalid_schema_text
line_mapping_error
glossary_violation
required_term_drop
oververbose_translation
suspicious_added_name
suspicious_added_gender
suspicious_added_speaker
excessive_explanation
repeated_text
unbalanced_quotes_or_brackets
html_or_markdown_noise
```

Hard failures:

```text
empty_output for translatable dialogue
assistant_chatter
line_mapping_error
invalid schema text
forbidden pattern
Japanese leakage above threshold
source copied verbatim as target
glossary violation for required terms
```

Warnings, not hard failures:

```text
slightly long output
minor punctuation weirdness
possible but not certain added name
possible style mismatch
```

Japanese leakage default:

```text
hard fail if target contains Japanese chars and case.source_type is not allowed to preserve Japanese
warning if a glossary term intentionally allows romanization or Japanese preservation
```

Oververbose proxy:

```text
if len(target_words) > max(18, 3.2 * estimated_source_units)
```

Do not overfit this proxy. It is one signal only.

---

## Glossary checks

Implement in:

```text
translation_quality_autoresearch/scorers/glossary_checks.py
```

Load glossary from:

```text
benchmark/glossary.json
```

Support the existing project glossary style where possible:

```json
{
  "source_replacements": [
    {"source": "あじと", "target": "アジト"}
  ],
  "exact_phrases": {
    "ただいま": "I'm home."
  },
  "target_replacements": [
    {"source": "Ajit", "target": "hideout", "when_source_contains": "アジト"}
  ],
  "required_terms": [
    {
      "source_contains": "先輩",
      "target_any": ["senpai", "upperclassman"],
      "severity": "major"
    }
  ]
}
```

Add harness-specific support:

```json
{
  "names": [
    {
      "source": "花子",
      "target": "Hanako",
      "forbid": ["Hannah", "Flower Child"]
    }
  ],
  "terms": [
    {
      "source": "アジト",
      "target": "hideout",
      "forbid": ["Ajit", "base camp"]
    }
  ]
}
```

Glossary scorer output:

```json
{
  "violations": [
    {
      "type": "required_term_missing",
      "source": "アジト",
      "expected": ["hideout"],
      "severity": "major"
    }
  ],
  "hard_fail": true
}
```

---

## MQM-style judge

Implement in:

```text
translation_quality_autoresearch/scorers/mqm_judge.py
```

The judge must be optional. If no judge provider is configured, return:

```json
{"enabled": false, "errors": [], "critical_errors": 0, "major_errors": 0, "minor_errors": 0}
```

Support providers:

```text
none
local_qwen
openai_optional
```

Default is:

```text
none
```

Do not require an API key. Do not call network APIs unless the user explicitly configures the provider.

### MQM error schema

Each MQM judge output must be strict JSON:

```json
{
  "case_id": "seed_clean_dialogue_001",
  "candidate_id": "seed_clean_dialogue_001__qwen_page",
  "overall": "good",
  "errors": [
    {
      "category": "accuracy",
      "subcategory": "addition",
      "severity": "major",
      "source_span": "",
      "target_span": "John",
      "explanation": "Adds a name not present in source or context."
    }
  ],
  "summary": "Candidate is mostly faithful but adds an unsupported name."
}
```

Allowed categories:

```text
accuracy
fluency
terminology
style
locale
format
safety
```

Allowed subcategories:

```text
accuracy.omission
accuracy.addition
accuracy.mistranslation
accuracy.hallucinated_speaker
accuracy.hallucinated_name
accuracy.wrong_polarity
accuracy.wrong_relationship
accuracy.wrong_honorific
fluency.grammar
fluency.unnatural_english
fluency.awkward_literalism
terminology.glossary_violation
terminology.inconsistent_name
style.too_formal
style.too_verbose
style.over_explained
style.not_manga_dialogue
locale.culturally_misleading
format.assistant_chatter
format.json_leakage
format.line_mapping_error
format.japanese_leakage
safety.refusal_or_policy_chatter
```

Severity:

```text
critical
major
minor
```

Severity guidance:

```text
critical: changes the meaning enough that the reader misunderstands the scene, adds unsupported facts/names/speakers, or emits assistant chatter/refusal.
major: clear meaning error, omission, terminology failure, or serious unnaturalness.
minor: small fluency/style issue that does not change meaning.
```

### Judge prompt requirements

The judge prompt in `prompts/mqm_judge_prompt.txt` must instruct the judge:

- Evaluate faithfulness first.
- Do not reward fluent hallucinations.
- Do not penalize valid paraphrases merely because they differ from the reference.
- Do not require word-for-word literal translation.
- For ambiguous Japanese, penalize unsupported specificity.
- Return JSON only.
- Never invent source facts.
- Use the exact severity/category labels.

---

## Pairwise judge

Implement in:

```text
translation_quality_autoresearch/scorers/pairwise_judge.py
```

The pairwise judge compares the selected candidate against a baseline candidate. It must be optional. It should support the same providers as the MQM judge.

Output schema:

```json
{
  "case_id": "seed_clean_dialogue_001",
  "baseline_candidate_id": "seed_clean_dialogue_001__baseline",
  "candidate_id": "seed_clean_dialogue_001__qwen_page",
  "winner": "candidate",
  "faithfulness_winner": "candidate",
  "naturalness_winner": "candidate",
  "glossary_winner": "tie",
  "brevity_winner": "tie",
  "reason_summary": "Candidate is equally faithful and more natural as dialogue."
}
```

Allowed winner values:

```text
baseline
candidate
tie
invalid
```

Use pairwise results as a bonus/penalty in the score, not as a hard gate unless a hard deterministic failure appears.

---

## Optional MT metrics

Implement in:

```text
translation_quality_autoresearch/scorers/comet_scorer.py
```

The scorer must gracefully degrade:

```text
If COMET is unavailable, skip and write null metrics.
If model weights are unavailable, skip and write null metrics.
If GPU is unavailable, allow CPU or skip based on config.
```

Do not download models implicitly unless the user passes:

```text
--allow-metric-downloads
```

Supported optional metrics:

```text
reference-based COMET when reference_translations exist
CometKiwi or other QE model when no reference exists
xCOMET if installed and configured
```

Output:

```json
{
  "comet": 0.82,
  "cometkiwi": null,
  "xcomet": null,
  "enabled_metrics": ["comet"],
  "skipped_metrics": ["cometkiwi", "xcomet"]
}
```

The composite score must not fail if all MT metrics are null.

---

## Manga style checks

Implement in:

```text
translation_quality_autoresearch/scorers/manga_style_checks.py
```

This should be heuristic and conservative.

Check:

```text
length ratio
word count
excessive explanation
translator notes
repeated punctuation spam
unnatural quote wrapping
unwanted speaker labels
unwanted stage directions
literal Japanese honorific handling when not desired
missing honorific when explicitly required
```

Do not aggressively penalize creative but faithful phrasing. Style checks should mostly catch obvious non-dialogue output.

Output:

```json
{
  "style_mismatch": false,
  "oververbose": false,
  "dialogue_naturalness_proxy": 0.82,
  "warnings": []
}
```

---

## Candidate scoring formula

Implement in:

```text
translation_quality_autoresearch/scorers/score_formula.py
```

Use lower-is-better candidate score:

```text
candidate_quality_score =
  100000 * hard_reject
+  30000 * critical_mqm_errors
+  12000 * major_mqm_errors
+   2500 * minor_mqm_errors
+  20000 * assistant_chatter
+  16000 * japanese_leakage
+  15000 * line_mapping_error
+  12000 * glossary_violation
+  10000 * hallucinated_name_or_speaker
+   9000 * forbidden_pattern
+   8000 * source_copied
+   7000 * empty_output
+   5000 * required_term_drop
+   3500 * oververbose
+   2500 * style_mismatch
+   1000 * repeated_text
+    600 * deterministic_warning_count
+    300 * latency_seconds
+    100 * cost_proxy
-   1500 * pairwise_win_bonus
-   1000 * normalized_comet_gain
```

Rules:

- If MQM is disabled, MQM counts are zero.
- If pairwise is disabled, `pairwise_win_bonus = 0`.
- If COMET is unavailable, `normalized_comet_gain = 0`.
- Clamp metric bonuses so a candidate with hard failures cannot win due to high COMET/pairwise scores.
- A hard-rejected candidate must not be selected if any non-hard-rejected candidate exists.

---

## Run-level scoring formula

The primary run score must be named:

```text
translation_quality_score
```

Compute from final selected case decisions:

```text
translation_quality_score =
  20000 * critical_mqm_error_rate
+ 12000 * hallucinated_addition_rate
+ 10000 * meaning_omission_rate
+  9000 * mistranslation_rate
+  8000 * wrong_name_or_speaker_rate
+  7000 * glossary_violation_rate
+  6000 * japanese_leakage_rate
+  5000 * assistant_chatter_rate
+  4000 * invalid_schema_rate
+  3000 * bad_line_mapping_rate
+  2500 * unnatural_english_rate
+  2000 * style_mismatch_rate
+  1500 * oververbose_translation_rate
+  1000 * false_reject_rate
+   500 * unnecessary_repair_rate
+   200 * mean_latency_seconds
+   100 * mean_cost_proxy
-   750 * pairwise_win_rate_vs_baseline
-   500 * mean_mt_metric_gain
```

Definitions:

```text
critical_mqm_error_rate = cases with >=1 critical MQM error / cases_total
hallucinated_addition_rate = cases with hallucinated addition/name/speaker/fact / cases_total
meaning_omission_rate = cases with MQM accuracy.omission major/critical / cases_total
mistranslation_rate = cases with MQM accuracy.mistranslation major/critical / cases_total
wrong_name_or_speaker_rate = cases with hallucinated_name or hallucinated_speaker / cases_total
glossary_violation_rate = selected outputs with glossary hard violation / cases_total
japanese_leakage_rate = selected outputs with Japanese leakage hard fail / cases_total
assistant_chatter_rate = selected outputs with assistant chatter / cases_total
invalid_schema_rate = selected outputs with JSON/schema leakage or invalid agent output / cases_total
bad_line_mapping_rate = source_hash mismatch or case_id mismatch / cases_total
unnatural_english_rate = fluency major/critical or style proxy hard issue / cases_total
style_mismatch_rate = style major/critical or explicit style mismatch / cases_total
oververbose_translation_rate = selected outputs exceeding verbosity threshold / cases_total
false_reject_rate = translatable cases where all candidates hard-rejected but at least one reference/frozen good candidate exists
unnecessary_repair_rate = repair candidate selected when non-repair candidate had equal-or-better score within tolerance
pairwise_win_rate_vs_baseline = pairwise candidate wins / judged pairwise cases
mean_mt_metric_gain = normalized improvement over baseline, or 0 when unavailable
```

Hard reject a run if:

```text
assistant_chatter_rate > 0
bad_line_mapping_rate > 0
critical_mqm_error_rate increases versus baseline
japanese_leakage_rate increases versus baseline
glossary_violation_rate increases versus baseline
case count differs from benchmark case count
any case_id/source_hash mismatch is detected
scorer or benchmark files were modified in the same experiment
```

---

## Result log

Create append-only TSV:

```text
translation_quality_autoresearch/results/translation_quality_results.tsv
```

Header:

```text
run_id
timestamp
commit
experiment_name
mode
changed_files
tests_ok
benchmark_valid
cases_total
candidate_count
selected_agent_counts
translation_quality_score
baseline_score
delta_score
critical_mqm_error_rate
major_mqm_error_rate
minor_mqm_error_rate
hallucinated_addition_rate
meaning_omission_rate
mistranslation_rate
wrong_name_or_speaker_rate
glossary_violation_rate
japanese_leakage_rate
assistant_chatter_rate
invalid_schema_rate
bad_line_mapping_rate
unnatural_english_rate
style_mismatch_rate
oververbose_translation_rate
false_reject_rate
unnecessary_repair_rate
pairwise_win_rate_vs_baseline
mean_comet_score
mean_cometkiwi_score
mean_xcomet_score
mean_mt_metric_gain
repair_rate
qwen_calls
cat_calls
opus_calls
vision_calls
mean_latency_seconds
p95_latency_seconds
mean_cost_proxy
hard_reject
kept
notes
```

Rules:

- Create the file with header if missing.
- Append one row per run.
- Never overwrite old rows.
- Do not mark `kept=true` automatically unless an explicit `--mark-kept` flag is passed.
- Write `hard_reject=true` when hard reject conditions are triggered.
- `changed_files` can be populated from `git diff --name-only HEAD` when available; otherwise write `unknown`.
- `commit` can be populated from `git rev-parse HEAD`; otherwise write `unknown`.

Implement log appending in:

```text
translation_quality_autoresearch/common/result_log.py
```

---

## Per-run outputs

Each evaluation run must write:

```text
translation_quality_autoresearch/runs/<run_id>/
  run_config.json
  summary.json
  summary.md
  candidate_scores.jsonl
  case_decisions.jsonl
  tag_breakdown.json
  failure_table.tsv
  hard_failures.jsonl
  best_examples.jsonl
  worst_examples.jsonl
  judge_outputs.jsonl
  metrics_by_agent.json
  diff_vs_baseline.json
```

`summary.json` must include:

```json
{
  "run_id": "baseline_replay",
  "mode": "replay",
  "cases_total": 12,
  "candidate_count": 36,
  "translation_quality_score": 123.4,
  "hard_reject": false,
  "rates": {
    "assistant_chatter_rate": 0.0,
    "japanese_leakage_rate": 0.0
  },
  "selected_agent_counts": {
    "qwen_page": 5,
    "cat": 4,
    "opus": 3
  },
  "tag_breakdown": {},
  "notes": []
}
```

`summary.md` must be human-readable and include:

```text
Top regressions
Top improvements
Worst cases
Hard failures
Selected agent distribution
Tag-level score slices
Commands to reproduce
```

---

## Replay mode

Replay mode evaluates frozen candidate outputs and reranks them. It must not call external models or project translator functions.

Command:

```bash
python translation_quality_autoresearch/scripts/eval_translation_quality.py \
  --mode replay \
  --cases translation_quality_autoresearch/benchmark/cases.jsonl \
  --references translation_quality_autoresearch/benchmark/references.jsonl \
  --frozen-outputs translation_quality_autoresearch/benchmark/frozen_agent_outputs.jsonl \
  --glossary translation_quality_autoresearch/benchmark/glossary.json \
  --output translation_quality_autoresearch/runs/baseline_replay \
  --results translation_quality_autoresearch/results/translation_quality_results.tsv \
  --run-id baseline_replay \
  --overwrite-output
```

Replay mode is used to test:

```text
schemas
scoring
hard checks
reranking
summaries
result logging
benchmark validation
```

Replay mode must be deterministic.

---

## Live-trace mode

Live-trace mode evaluates candidates generated by `run_translation_agents.py`.

Command:

```bash
python translation_quality_autoresearch/scripts/eval_translation_quality.py \
  --mode live-trace \
  --cases translation_quality_autoresearch/benchmark/cases.jsonl \
  --references translation_quality_autoresearch/benchmark/references.jsonl \
  --traces translation_quality_autoresearch/runs/live_candidates/agent_traces.jsonl \
  --glossary translation_quality_autoresearch/benchmark/glossary.json \
  --output translation_quality_autoresearch/runs/live_eval \
  --results translation_quality_autoresearch/results/translation_quality_results.tsv \
  --run-id live_eval \
  --overwrite-output
```

Live-trace mode must still be deterministic once traces exist. It must not re-call models during evaluation.

---

## Live agent generation

Implement:

```text
translation_quality_autoresearch/scripts/run_translation_agents.py
```

Purpose:

```text
Read TranslationCase records.
Call selected translator agents.
Write AgentTraceRecord JSONL.
```

Supported agents:

```text
opus
madlad
argos
cat
qwen_block
qwen_page
qwen_repair
qwen_critic_repair
qwen_vision_context_optional
```

Minimum implementation:

```text
opus if available
cat if available/configured
qwen_block if available/configured
qwen_page if available/configured
```

CLI options:

```text
--cases PATH_OR_LIST
--agents opus,qwen_block,qwen_page,cat
--output PATH
--glossary PATH
--config translation_quality_autoresearch/config/default_live_agents.json
--qwen-model PATH
--qwen-critic-model PATH
--qwen-fallback-model PATH
--cat-model PATH
--temperature FLOAT
--max-tokens INT
--timeout-seconds INT
--resume
--limit N
--case-id CASE_ID
--dry-run
```

Rules:

- Do not render images.
- Do not call detection/OCR.
- Do not mutate benchmark files.
- Preserve `case_id` and `source_hash` in every trace.
- Use context fields from the case where supported.
- If an agent is unavailable, record a skipped-agent warning but do not crash the whole run unless `--strict-agents` is passed.
- Store prompts and prompt hashes.
- Store latency and a simple cost proxy.
- Do not hide raw outputs; store them in traces.

Implementation advice:

- Use dynamic imports from `manga_local_translator` through `common/project_imports.py`.
- Inspect actual translator class/function signatures before coding.
- Prefer existing project APIs rather than duplicating model-loading logic.
- If direct import is hard, add a simple adapter layer inside the harness, not inside the main project.

---

## Reranker

Implement:

```text
translation_quality_autoresearch/scripts/rerank_candidates.py
```

and reusable logic in:

```text
translation_quality_autoresearch/common/metrics.py
```

The evaluator may call reranking internally, but the standalone script is useful for debugging.

Reranking policy:

1. Remove hard-rejected candidates if any non-hard-rejected candidate exists.
2. Rank by `candidate_quality_score` ascending.
3. Tie-break by fewer critical/major errors.
4. Tie-break by fewer deterministic warnings.
5. Tie-break by better glossary compliance.
6. Tie-break by shorter/natural renderable text, within reason.
7. Tie-break by lower latency/cost.
8. Prefer baseline candidate only if it is equal or better within tolerance.

Standalone command:

```bash
python translation_quality_autoresearch/scripts/rerank_candidates.py \
  --candidate-scores translation_quality_autoresearch/runs/baseline_replay/candidate_scores.jsonl \
  --output translation_quality_autoresearch/runs/baseline_replay/reranked.jsonl
```

---

## Benchmark builder

Implement:

```text
translation_quality_autoresearch/scripts/build_cases_from_quality_runs.py
```

Purpose:

```text
Convert existing debug/quality-run outputs into benchmark candidate cases.
```

It should search inputs such as:

```text
quality-runs/**
.manga-work/**
*.ocr.json
```

But it must write generated cases only inside:

```text
translation_quality_autoresearch/benchmark/generated/
```

Create the `generated/` subfolder only when needed.

CLI:

```bash
python translation_quality_autoresearch/scripts/build_cases_from_quality_runs.py \
  --input quality-runs \
  --output translation_quality_autoresearch/benchmark/generated/generated_cases.jsonl \
  --min-source-chars 1 \
  --max-cases 200 \
  --include-context \
  --dry-run
```

Rules:

- Do not overwrite `benchmark/cases.jsonl` unless `--overwrite` is passed.
- Generated cases are candidates for human review, not automatically trusted gold.
- Include source hashes.
- Include source text, page id, group id, context before/after, OCR risk if available, and translation output if available.
- Mark generated cases with:

```json
"metadata": {"generated_from_quality_run": true, "needs_human_review": true}
```

---

## Human gold cases

`benchmark/human_gold.jsonl` should allow stricter gates for a small manually reviewed set.

Schema:

```json
{
  "case_id": "gold_001",
  "source_text": "ただいま",
  "source_hash": "sha256:...",
  "gold_rating": {
    "must_select_one_of": ["I'm home."],
    "forbid": ["I am back now with information."],
    "notes": "Common homecoming phrase."
  },
  "severity_if_failed": "critical"
}
```

Evaluation behavior:

- If a human-gold case fails a critical rule, mark run `hard_reject=true`.
- Do not require exact string match unless `must_select_exact` is provided.
- Prefer meaning and constraints over exact surface form.

---

## Prompt files

Create prompt files under:

```text
translation_quality_autoresearch/prompts/
```

These are harness prompts for live generation/judging experiments. They should be editable in future prompt-search runs without changing the main project.

### candidate_translation_prompt.txt

Should instruct:

```text
Translate Japanese manga dialogue to natural English.
Preserve meaning.
Do not add names, speakers, genders, facts, or explanations not present in source/context.
Use context only to resolve ambiguity, not to invent plot.
Return only the English translation.
No labels, no JSON, no commentary.
```

### page_context_translation_prompt.txt

Should include:

```text
source line
previous lines
next lines
speaker hint if available
glossary terms
style notes
```

### critic_prompt.txt

Should check:

```text
faithfulness
omission
addition
wrong speaker/name/gender
glossary
Japanese leakage
assistant chatter
naturalness
```

### repair_prompt.txt

Should instruct:

```text
Repair only the listed problems.
Do not rewrite good content unnecessarily.
Do not add new facts.
Return only the repaired translation.
```

### mqm_judge_prompt.txt

Use the MQM schema described above.

### pairwise_judge_prompt.txt

Compare candidate vs baseline on:

```text
faithfulness
naturalness
glossary adherence
brevity/renderability
lack of hallucination
```

---

## Optimizers

Create stubs that are safe and optional.

### failure_reflection.py

Purpose:

```text
Read worst failures from a run and produce human-readable suggestions.
```

Command:

```bash
python translation_quality_autoresearch/optimizers/failure_reflection.py \
  --run translation_quality_autoresearch/runs/baseline_replay \
  --output translation_quality_autoresearch/runs/baseline_replay/failure_reflection.md
```

It may use no model by default. It should summarize:

```text
most common hard failures
most common MQM categories
worst tags
agents causing most failures
candidate examples
possible prompt/rule changes
```

### prompt_search.py

Purpose:

```text
Try manually specified prompt variants against live or replay traces.
```

Do not implement complex optimization at first. The initial version can run variants from a folder:

```text
translation_quality_autoresearch/prompts/variants/<variant_name>/
```

and write one result row per variant.

### dspy_mipro_adapter.py

Stub only at first.

Behavior:

- If DSPy is not installed, print a clear message and exit 0 unless `--strict` is passed.
- Do not require DSPy for baseline harness operation.
- Include comments explaining that MIPROv2 can later optimize instructions/few-shot examples against `translation_quality_score`.

### gepa_adapter.py

Stub only at first.

Behavior:

- If DSPy/GEPA is not installed, print a clear message and exit 0 unless `--strict` is passed.
- Do not require it for baseline harness operation.
- Include comments explaining that GEPA-style optimization should use full execution traces and textual feedback.

---

## Config files

### config/default_eval.json

```json
{
  "mqm_judge_provider": "none",
  "pairwise_judge_provider": "none",
  "enable_comet": false,
  "enable_cometkiwi": false,
  "enable_xcomet": false,
  "allow_metric_downloads": false,
  "baseline_agent": "baseline",
  "hard_fail_japanese_leakage_ratio": 0.02,
  "oververbose_word_ratio": 3.2,
  "max_target_words_default": 28,
  "selection_tolerance": 0.001,
  "write_judge_outputs": true
}
```

### config/default_live_agents.json

```json
{
  "agents": ["opus", "qwen_block", "qwen_page", "cat"],
  "temperature": 0.2,
  "max_tokens": 256,
  "timeout_seconds": 120,
  "qwen_model": null,
  "qwen_critic_model": null,
  "qwen_fallback_model": null,
  "cat_model": null,
  "use_page_context": true,
  "use_glossary": true,
  "store_raw_outputs": true
}
```

### config/scoring_weights.json

Mirror the run-level and candidate-level weights. `score_formula.py` should load this file by default, but also contain defaults if the file is missing.

### config/judge_rubric_mqm.json

Include category/subcategory/severity definitions.

### config/reranker_weights.json

Include tie-breaker and selection weights.

---

## Validation script

Implement:

```text
translation_quality_autoresearch/scripts/validate_benchmark.py
```

Checks:

```text
all JSONL files parse
case_id unique
source_text non-empty
source_hash matches normalized source_text
references only reference known case_id
frozen outputs only reference known case_id
all candidates have candidate_id, agent, text
no duplicate candidate_id within case
human_gold only references known case_id or contains full source fields
forbidden_patterns is list
must_preserve is list
context_before/context_after are lists
```

Command:

```bash
python translation_quality_autoresearch/scripts/validate_benchmark.py \
  --benchmark translation_quality_autoresearch/benchmark
```

Exit with nonzero status on validation failure.

---

## Unit tests

Implement unit tests under:

```text
translation_quality_autoresearch/tests/
```

Minimum tests:

### test_schemas.py

- Loads seed cases.
- Validates required fields.
- Confirms hash generation is stable.
- Confirms source hash mismatch is caught.

### test_deterministic_checks.py

Cases:

```text
assistant chatter hard-fails
Japanese leakage hard-fails
empty output hard-fails
source copied hard-fails
forbidden pattern hard-fails
glossary required term missing hard-fails
oververbose output warns or penalizes
normal good candidate passes
```

### test_score_formula.py

- Hard-rejected candidate loses to normal candidate.
- Critical MQM errors dominate style bonuses.
- COMET/pairwise bonuses cannot rescue hard failures.
- Run-level score is lower for fewer failures.

### test_result_log.py

- Creates TSV with header.
- Appends rows.
- Does not overwrite existing rows.

### test_reranker.py

- Selects good candidate from seed frozen outputs.
- Rejects assistant chatter even if text is fluent.
- Rejects Japanese leakage.
- Tie-breaks by cost only after quality ties.

---

## Implementation order

Follow this order exactly:

### Phase 1: skeleton and seed data

1. Create `translation_quality_autoresearch/` tree.
2. Write README and PROGRAM.
3. Add seed benchmark cases and frozen outputs.
4. Add config JSON files.
5. Add `.gitkeep` files.

### Phase 2: schema and validation

1. Implement `schemas.py`.
2. Implement `io_utils.py`.
3. Implement `text_normalize.py` and hashing.
4. Implement `validate_benchmark.py`.
5. Add schema tests.

### Phase 3: deterministic replay evaluation

1. Implement deterministic checks.
2. Implement glossary checks.
3. Implement style checks.
4. Implement candidate score formula.
5. Implement reranker.
6. Implement result log appending.
7. Implement `eval_translation_quality.py` replay mode.
8. Add tests.

At the end of Phase 3, this command must work:

```bash
python translation_quality_autoresearch/scripts/eval_translation_quality.py \
  --mode replay \
  --cases translation_quality_autoresearch/benchmark/cases.jsonl \
  --references translation_quality_autoresearch/benchmark/references.jsonl \
  --frozen-outputs translation_quality_autoresearch/benchmark/frozen_agent_outputs.jsonl \
  --glossary translation_quality_autoresearch/benchmark/glossary.json \
  --output translation_quality_autoresearch/runs/baseline_replay \
  --results translation_quality_autoresearch/results/translation_quality_results.tsv \
  --run-id baseline_replay \
  --overwrite-output
```

### Phase 4: live trace generation

1. Implement `project_imports.py`.
2. Implement agent adapters in `run_translation_agents.py`.
3. Add `--dry-run` and `--strict-agents`.
4. Write traces without evaluating them.
5. Then evaluate traces with live-trace mode.

### Phase 5: optional judges and MT metrics

1. Implement MQM judge with provider `none` first.
2. Add local Qwen provider if project APIs make it straightforward.
3. Add optional OpenAI provider only behind explicit config and env-var checks.
4. Add optional COMET scorer that skips cleanly when missing.
5. Add pairwise judge.

### Phase 6: optimization stubs

1. Implement failure reflection.
2. Add prompt variant search.
3. Add DSPy/MIPROv2 stub.
4. Add GEPA stub.

---

## eval_translation_quality.py details

CLI:

```text
--mode replay|live-trace
--cases PATH_OR_LIST
--references PATH
--frozen-outputs PATH
--traces PATH
--glossary PATH
--config PATH
--weights PATH
--output PATH
--results PATH
--run-id TEXT
--baseline-summary PATH
--experiment-name TEXT
--notes TEXT
--limit N
--tag TAG
--case-id CASE_ID
--overwrite-output
--append-results / --no-append-results
--mark-kept
--mqm-judge-provider none|local_qwen|openai_optional
--pairwise-judge-provider none|local_qwen|openai_optional
--enable-comet
--enable-cometkiwi
--enable-xcomet
--allow-metric-downloads
```

Behavior:

1. Load config.
2. Load cases.
3. Load references.
4. Load glossary.
5. Load candidate source:
   - replay: frozen outputs
   - live-trace: traces
6. Validate case IDs and hashes.
7. Run deterministic checks per candidate.
8. Run optional style/glossary checks.
9. Run optional MQM judge.
10. Run optional MT metrics.
11. Compute candidate scores.
12. Rerank per case.
13. Select final candidate per case.
14. Compute run-level metrics.
15. Compute `translation_quality_score`.
16. Write per-run outputs.
17. Append result TSV unless disabled.
18. Exit nonzero only on infrastructure/validation failures, not normal bad scores.

Do not use model judges or COMET by default.

---

## compare_runs.py details

Implement:

```text
translation_quality_autoresearch/scripts/compare_runs.py
```

Command:

```bash
python translation_quality_autoresearch/scripts/compare_runs.py \
  --baseline translation_quality_autoresearch/runs/baseline_replay/summary.json \
  --candidate translation_quality_autoresearch/runs/current/summary.json \
  --output translation_quality_autoresearch/runs/current/compare_vs_baseline.md
```

Report:

```text
score delta
hard reject status
metric deltas
tag-level deltas
agent selection changes
new hard failures
resolved hard failures
examples improved
examples regressed
recommend keep/revert
```

Keep/revert recommendation:

```text
KEEP if score improves and no hard reject metric regresses.
REVERT if score worsens, hard reject is true, or critical rates increase.
MANUAL_REVIEW if score improves but important submetrics are mixed.
```

---

## summarize_run.py details

Implement:

```text
translation_quality_autoresearch/scripts/summarize_run.py
```

Command:

```bash
python translation_quality_autoresearch/scripts/summarize_run.py \
  --run translation_quality_autoresearch/runs/baseline_replay
```

It should print a concise terminal summary and optionally rewrite `summary.md`.

---

## optional_requirements.txt

Create:

```text
translation_quality_autoresearch/optional_requirements.txt
```

Suggested content:

```text
# Optional only. The baseline harness must run without these.
# Install manually if you want neural MT metrics or prompt optimization.
unbabel-comet
# dspy
# inspect-ai
# promptfoo
```

Do not add these to root `requirements.txt` during initial setup.

---

## Acceptance criteria

The implementation is complete when all of these pass:

```bash
python translation_quality_autoresearch/scripts/validate_benchmark.py \
  --benchmark translation_quality_autoresearch/benchmark

python -m unittest discover -s translation_quality_autoresearch/tests

python translation_quality_autoresearch/scripts/eval_translation_quality.py \
  --mode replay \
  --cases translation_quality_autoresearch/benchmark/cases.jsonl \
  --references translation_quality_autoresearch/benchmark/references.jsonl \
  --frozen-outputs translation_quality_autoresearch/benchmark/frozen_agent_outputs.jsonl \
  --glossary translation_quality_autoresearch/benchmark/glossary.json \
  --output translation_quality_autoresearch/runs/baseline_replay \
  --results translation_quality_autoresearch/results/translation_quality_results.tsv \
  --run-id baseline_replay \
  --overwrite-output
```

And these files exist:

```text
translation_quality_autoresearch/runs/baseline_replay/summary.json
translation_quality_autoresearch/runs/baseline_replay/summary.md
translation_quality_autoresearch/runs/baseline_replay/candidate_scores.jsonl
translation_quality_autoresearch/runs/baseline_replay/case_decisions.jsonl
translation_quality_autoresearch/runs/baseline_replay/failure_table.tsv
translation_quality_autoresearch/results/translation_quality_results.tsv
```

The seed benchmark must demonstrate that:

```text
good candidates beat bad candidates
assistant chatter is hard-rejected
Japanese leakage is hard-rejected
glossary violations are penalized
hard failures cannot be rescued by bonuses
result logs append one row
```

---

## Guardrails for future autoresearch experiments

Once the harness is implemented, future experiments may optimize:

```text
translation prompts
critic prompts
repair prompts
few-shot examples
glossary injection format
candidate reranker weights
repair trigger thresholds
agent selection policy
Qwen page-context behavior
CAT/Qwen fallback policy
```

Future experiments must not optimize by changing:

```text
benchmark cases
reference records
human-gold records
frozen outputs, unless creating a new explicitly named benchmark version
scoring formulas
hard failure rules
result TSV history
unit tests
```

If the benchmark or scorer must change, create a new benchmark version and record it explicitly, for example:

```text
translation_quality_autoresearch/benchmark_v2/
translation_quality_autoresearch/results/translation_quality_results_v2.tsv
```

Do not mix scores from different benchmark versions.

---

## Notes on recent harness ideas to incorporate

The design intentionally borrows these modern harness patterns:

- Custom application-specific eval datasets rather than generic public benchmarks.
- Deterministic checks plus model-graded evaluation, not model judging alone.
- Agent trace capture, not just final output scoring.
- Pairwise baseline comparison.
- MQM-style translation error categories and severity weighting.
- Optional MT-specific neural metrics such as COMET/CometKiwi/xCOMET.
- Replay mode for cheap deterministic scoring.
- Live mode for prompt/model changes.
- Prompt/program optimization stubs for later DSPy/MIPRO/GEPA-style work.

Do not overbuild the first version. The baseline harness should work with only Python standard library plus the existing project environment.

---

## Final reminder to Codex

Keep this harness apart from the main project.

The root folder must be:

```text
translation_quality_autoresearch/
```

The harness may read and import the project, but it must not become part of the main translation pipeline unless a later explicit task asks for integration.

The first deliverable is a reproducible replay-mode evaluation loop with seed benchmark data, deterministic scoring, reranking, run summaries, and append-only result logging.
