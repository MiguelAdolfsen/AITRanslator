# Benchmark Readiness

Status: `NEEDS_REVIEWED_BENCHMARK_EXPANSION`

Manual Codex handoff status: `READY_FOR_MACHINE_ASSISTED_AGENT_HANDOFF` for `benchmark_v1/`.

The checked-in benchmark is enough for harness validation and agent workflow dry runs. It is not enough for autonomous optimization of translator prompts, rules, or reranking policy.

## Current coverage

The current benchmark files contain 14 total case records across:

- `benchmark/cases.jsonl`: 12 cases.
- `benchmark/adversarial_cases.jsonl`: 1 case.
- `benchmark/page_context_cases.jsonl`: 1 case.
- `benchmark/seed_cases.jsonl`: 0 cases.

Coverage summary:

- References: 14 of 14 selected cases.
- Human-gold gates: 1 case.
- Context cases: 6 cases.
- Glossary cases: 1 case.
- Must-preserve cases: 3 cases.
- Multi-line source cases: 0 cases.
- Source types: 13 dialogue, 1 SFX.
- OCR risk: 13 clean, 1 noisy salvageable.

## Main gaps

- Total reviewed cases are short by at least 46 for a 60-case v1 minimum.
- SFX/reaction cases are underrepresented.
- No multi-line or line-mapping cases are present.
- No real reviewed holdout set exists.
- Human-gold gates cover only one critical OCR-noise case.
- Glossary, honorific, ambiguous-speaker, page-context, and noisy-OCR coverage are each too shallow.

## Required before autonomous optimization

Create a reviewed benchmark version with:

- 60-100 reviewed dev cases.
- A separate holdout split with 20-30% as many cases as the dev set.
- References for every case.
- Frozen candidates or reproducible live traces for every case.
- Human-gold gates for critical regressions.
- Separate result log for the benchmark version.

See `benchmark/VERSIONING.md` for the target distribution.

## Manual Codex v1

`benchmark_v1/` contains 60 mined cases with Codex-manual references and frozen baseline outputs from `metadata.draft_machine_translation`. `benchmark_holdout_v1/` contains 15 additional mined cases for final gating. Use these for early agent handoff only; they do not replace a native-reviewed human-gold benchmark.

## Coverage command

Run this after adding or changing benchmark files:

```powershell
.\.venv\Scripts\python.exe translation_quality_autoresearch\scripts\report_benchmark_coverage.py `
  --cases translation_quality_autoresearch\benchmark\cases.jsonl,translation_quality_autoresearch\benchmark\adversarial_cases.jsonl,translation_quality_autoresearch\benchmark\page_context_cases.jsonl `
  --output translation_quality_autoresearch\runs\benchmark_coverage.md
```

Use `--strict` when a CI or release check should fail until benchmark targets are met.
