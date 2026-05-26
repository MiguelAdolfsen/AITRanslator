# Source Extraction Autoresearch Program

Objective: minimize `source_extraction_score`.

Minimize missed, false, garbled, wrongly grouped, and wrongly ordered Japanese source blocks before translation.

Scope: detection, OCR crop preparation, OCR acceptance, text filtering, grouping, orientation, and reading order only.

Frozen benchmark rule: during optimization, do not modify benchmark cases, labels, scoring, templates, result-log schema, or existing result rows.

Anti-overfit rule: do not improve benchmark scores by matching exact frozen-page text, exact OCR mistakes, page numbers, source folders, comic titles, character names, known benchmark coordinates, or one-off benchmark layouts. Extraction rules must be explainable by generic OCR/detector evidence such as script class, punctuation class, confidence, orientation, line geometry, connected text-region structure, crop quality, or repeated behavior across unrelated pages.

Do not add source-specific OCR rewrite maps in source extraction code. Examples of forbidden patterns:

```text
exact Japanese phrase -> corrected Japanese phrase
exact OCR-garbled string -> expected benchmark label
page-id/filename/folder checks
coordinate-only filters that suppress text because it appears at a known page edge
single-character/name/SFX suppressions that depend on one benchmark failure
```

Allowed cleanup must be generic and auditable. For example, whitespace normalization, punctuation normalization, confidence thresholds, shape/script-based SFX filters, metadata filters based on actual metadata/credit language, and crop changes based on CTD line polygons are acceptable when covered by tests and real-page benchmarks.

Required setup:

```powershell
.\.venv\Scripts\python.exe source_extraction_autoresearch\scripts\generate_synthetic_cases.py `
  --output source_extraction_autoresearch\benchmarks\synthetic
```

Required fixture validation:

```powershell
.\.venv\Scripts\python.exe source_extraction_autoresearch\scripts\validate_fixtures.py `
  --benchmark source_extraction_autoresearch\benchmarks\synthetic

.\.venv\Scripts\python.exe source_extraction_autoresearch\scripts\validate_fixtures.py `
  --benchmark source_extraction_autoresearch\benchmarks\cases\local_real_diverse_v2_frozen
```

Required unit tests:

```powershell
.\.testing\run_tests.ps1
```

Required benchmark:

```powershell
.\.venv\Scripts\python.exe source_extraction_autoresearch\scripts\eval_source_extraction.py `
  --benchmark source_extraction_autoresearch\benchmarks\synthetic `
  --output source_extraction_autoresearch\runs\current `
  --results source_extraction_autoresearch\results\results.tsv `
  --run-id current `
  --run-tests
```

Required local real-page benchmark after the synthetic smoke benchmark passes:

```powershell
.\.venv\Scripts\python.exe source_extraction_autoresearch\scripts\eval_source_extraction.py `
  --benchmark source_extraction_autoresearch\benchmarks\cases\local_real_diverse_v2_frozen `
  --output source_extraction_autoresearch\runs\local_real_diverse_v2_current `
  --results source_extraction_autoresearch\results\results.tsv `
  --run-id local_real_diverse_v2_current `
  --run-tests `
  --overwrite-output
```

Matrix gate command:

```powershell
.\.venv\Scripts\python.exe source_extraction_autoresearch\scripts\run_benchmark_matrix.py `
  --include-historical `
  --overwrite-output
```

Local real cases are ignored by git because they contain real manga pages. Synthetic is only a smoke/regression benchmark.

Current local real-case matrix:

```text
benchmarks/cases/local_real_diverse_v2_frozen/    primary broad real-page benchmark
benchmarks/cases/local_spy_short_v1_frozen/       historical v1 regression
benchmarks/cases/local_hard_pages_v1_frozen/      historical v1 hard-page regression
```

Use synthetic plus `local_real_diverse_v2_frozen/` as the required keep/revert gate. Run the two v1 frozen cases as historical regression checks when changing grouping, ordering, OCR acceptance, or filtering behavior that could regress older coverage.

Phase-1 editable project-code surface:

```text
manga_local_translator/source_extraction.py
manga_local_translator/detect_ocr.py
manga_local_translator/grouping.py
manga_local_translator/text_filter.py
```

Phase-2 editable surface, only after the phase-1 loop is stable:

```text
manga_local_translator/ctd_detector.py
manga_local_translator/config.py
manga_local_translator/pipeline.py
```

`pipeline.py` may be edited only for cleanly exposing source-extraction outputs or avoiding duplicated source-extraction work. Do not let this loop change translation, erase, or render behavior.

Forbidden edits during optimization:

```text
source_extraction_autoresearch/benchmarks/**
source_extraction_autoresearch/BENCHMARK_SCHEMA.md
source_extraction_autoresearch/SCORING.md
source_extraction_autoresearch/PROGRAM.md
source_extraction_autoresearch/scripts/eval_source_extraction.py
source_extraction_autoresearch/scripts/io_adapters.py
source_extraction_autoresearch/scripts/score.py
source_extraction_autoresearch/scripts/match_regions.py
source_extraction_autoresearch/scripts/normalize_text.py
source_extraction_autoresearch/scripts/validate_fixtures.py
source_extraction_autoresearch/scripts/summarize_results.py
source_extraction_autoresearch/scripts/report_failures.py
source_extraction_autoresearch/scripts/benchmark_coverage.py
source_extraction_autoresearch/scripts/rebuild_best_index.py
source_extraction_autoresearch/scripts/run_benchmark_matrix.py
source_extraction_autoresearch/templates/**
source_extraction_autoresearch/results/results.tsv
render_autoresearch/**
translation_routing_autoresearch/**
manga_local_translator/render.py
manga_local_translator/erase.py
manga_local_translator/translate.py
manga_local_translator/hf_translators.py
manga_local_translator/qwen_translator.py
manga_local_translator/qwen_validation.py
manga_local_translator/qwen_vision.py
manga_local_translator/vision_service.py
manga_local_translator/vision_artifact.py
.testing/tests/**
```

Keep a change only if:

```text
1. unit tests pass,
2. fixture validation passes,
3. source_extraction_score improves versus current best,
4. missed_dialogue_region_count does not increase,
5. destructive_false_positive_count does not increase,
6. mean_ocr_cer does not increase unless missed_dialogue_region_count improves,
7. overmerge_count and undermerge_count do not increase together,
8. reading_order_error_count does not increase by more than 2%,
9. orientation_error_count does not increase,
10. the improvement did not come from editing benchmark/scoring/logging files,
11. the improvement did not come from exact source-text rewrites, benchmark-specific phrase corrections, page/folder checks, or coordinate-only suppressions.
```

Otherwise revert the experiment change and append a failed result row. Every benchmark run appends exactly one row to `results/results.tsv`.
