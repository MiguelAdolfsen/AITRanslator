# Translation Quality Autoresearch Program

Goal: minimize `translation_quality_score` on the fixed benchmark.

This loop optimizes translation-agent quality only. It must not optimize detection, OCR, grouping, erase, rendering, GUI behavior, or benchmark scoring.

## Read first

Before every experiment, read:

- README.md
- translation_quality_autoresearch/README.md
- translation_quality_autoresearch/PROGRAM.md
- translation_quality_autoresearch/AGENT_START.md
- translation_quality_autoresearch/BENCHMARK_READINESS.md
- translation_quality_autoresearch/benchmark/VERSIONING.md
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
python translation_quality_autoresearch/scripts/report_benchmark_coverage.py --cases translation_quality_autoresearch/benchmark/cases.jsonl,translation_quality_autoresearch/benchmark/adversarial_cases.jsonl,translation_quality_autoresearch/benchmark/page_context_cases.jsonl
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

## Benchmark readiness

The seed benchmark is for smoke testing the harness and validating agent workflow only. Do not use it alone as evidence for product-quality autonomous optimization. Before broad prompt/rule search, create a reviewed benchmark version with at least 60-100 cases plus a holdout split, following `translation_quality_autoresearch/benchmark/VERSIONING.md`.

`benchmark_v1/` and `benchmark_holdout_v1/` are manual Codex handoff benchmarks. Agents may use `benchmark_v1/` for early iteration and `benchmark_holdout_v1/` as a final gate, but the references are not native-reviewed human-gold. Product-quality acceptance still requires human review or a later reviewed benchmark version.
