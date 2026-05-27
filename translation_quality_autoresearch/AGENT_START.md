# Translation Quality Agent Start

Use this file as the short startup contract for translation-quality agents.

## Current baseline

- Benchmark version: seed/smoke benchmark in `translation_quality_autoresearch/benchmark/`.
- Default cases: `translation_quality_autoresearch/benchmark/cases.jsonl`.
- Baseline run: `translation_quality_autoresearch/runs/baseline_replay/summary.json`.
- Baseline `translation_quality_score`: `3.016202`.
- Lower score is better.

The seed benchmark is intentionally small. It is suitable for harness smoke checks and workflow dry runs, not for fully autonomous optimization.

## Manual Codex handoff benchmarks

- Dev benchmark: `translation_quality_autoresearch/benchmark_v1/`.
- Holdout benchmark: `translation_quality_autoresearch/benchmark_holdout_v1/`.
- Reference policy: `manual_codex_translation`.
- Dev result log: `translation_quality_autoresearch/results/translation_quality_results_v1.tsv`.
- Holdout result log: `translation_quality_autoresearch/results/translation_quality_results_holdout_v1.tsv`.

These benchmarks are acceptable for early optimization-agent handoff. References were manually translated by Codex from decoded OCR source/context, but they are not native-reviewed human-gold/product-quality evidence. Keep changes only after improving dev results and passing the holdout gate without guardrail regressions.

## Goal

Minimize `translation_quality_score` for the fixed benchmark while preserving translation behavior outside this harness.

This loop optimizes only:

```text
validated OCR source blocks + context -> translator agents -> final English translation
```

It must not optimize detection, OCR, grouping, erase, rendering, inpainting, GUI behavior, or benchmark scoring.

## Read first

- `README.md`
- `translation_quality_autoresearch/README.md`
- `translation_quality_autoresearch/PROGRAM.md`
- `translation_quality_autoresearch/AGENT_START.md`
- `translation_quality_autoresearch/BENCHMARK_READINESS.md`
- `translation_quality_autoresearch/config/scoring_weights.json`
- `translation_quality_autoresearch/scorers/score_formula.py`
- `manga_local_translator/qwen_translator.py`
- `manga_local_translator/qwen_validation.py`
- `manga_local_translator/hf_translators.py`
- `manga_local_translator/quality_eval.py`
- `manga_local_translator/pipeline.py`

## Protected files

Do not edit these unless the explicit task is to improve the harness itself:

- `translation_quality_autoresearch/benchmark/*`
- `translation_quality_autoresearch/scorers/*`
- `translation_quality_autoresearch/scripts/eval_translation_quality.py`
- `translation_quality_autoresearch/results/*`
- `translation_quality_autoresearch/tests/*`

## Allowed edit targets

Primary translation-quality experiments may edit:

- `manga_local_translator/qwen_translator.py`
- `manga_local_translator/qwen_validation.py`
- `manga_local_translator/hf_translators.py`
- `translation_quality_autoresearch/prompts/*.txt`
- `translation_quality_autoresearch/config/reranker_weights.json`

Secondary targets require explicit approval:

- `manga_local_translator/pipeline.py`
- `manga_local_translator/config.py`
- `manga_local_translator/quality_eval.py`

## Required checks

Run these before keeping a change:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s translation_quality_autoresearch\tests

.\.venv\Scripts\python.exe translation_quality_autoresearch\scripts\validate_benchmark.py `
  --benchmark translation_quality_autoresearch\benchmark

.\.venv\Scripts\python.exe translation_quality_autoresearch\scripts\eval_translation_quality.py `
  --mode replay `
  --cases translation_quality_autoresearch\benchmark\cases.jsonl `
  --references translation_quality_autoresearch\benchmark\references.jsonl `
  --frozen-outputs translation_quality_autoresearch\benchmark\frozen_agent_outputs.jsonl `
  --glossary translation_quality_autoresearch\benchmark\glossary.json `
  --output translation_quality_autoresearch\runs\current `
  --results translation_quality_autoresearch\results\translation_quality_results.tsv `
  --run-id current `
  --overwrite-output
```

Use `--no-append-results` for local smoke checks that should not be recorded.

## Run artifacts

Every accepted experiment run must preserve:

```text
summary.json
case_results.jsonl
traces.jsonl
failures.jsonl
artifacts/run_manifest.json
```

Use `case_results.jsonl` for final selections, `traces.jsonl` for source/context/candidate/reranker details, and `failures.jsonl` for hard-check and warning review. `candidate_scores.jsonl` and `case_decisions.jsonl` are legacy compatibility outputs.

## Keep/revert policy

Keep a change only if:

- Tests pass.
- Benchmark validation passes.
- Evaluation completes.
- `translation_quality_score` improves versus the current best for the same benchmark version.
- Critical MQM errors do not increase.
- Japanese leakage does not increase.
- Assistant chatter does not increase.
- Glossary violations do not increase.
- Hard failures do not increase.
- Line/source mapping remains stable.

Otherwise revert the experiment and record the failure in run notes or the experiment log.

## Benchmark readiness

Before autonomous optimization, build a reviewed benchmark version with at least 60-100 cases and a holdout split. The current seed benchmark should not be treated as enough evidence for product-quality prompt/rule changes.
