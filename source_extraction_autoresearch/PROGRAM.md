# Source Extraction Autoresearch Program

Objective: minimize `source_extraction_score`.

Minimize missed, false, garbled, wrongly grouped, and wrongly ordered Japanese source blocks before translation.

Scope: detection, OCR crop preparation, OCR acceptance, text filtering, grouping, orientation, and reading order only.

Frozen benchmark rule: during optimization, do not modify benchmark cases, labels, scoring, templates, result-log schema, or existing result rows.

Required setup:

```powershell
.\.venv\Scripts\python.exe source_extraction_autoresearch\scripts\generate_synthetic_cases.py `
  --output source_extraction_autoresearch\benchmarks\synthetic
```

Required fixture validation:

```powershell
.\.venv\Scripts\python.exe source_extraction_autoresearch\scripts\validate_fixtures.py `
  --benchmark source_extraction_autoresearch\benchmarks\synthetic
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
  --tests-ok
```

Phase-1 editable project-code surface:

```text
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
source_extraction_autoresearch/scripts/eval_source_extraction.py
source_extraction_autoresearch/scripts/score.py
source_extraction_autoresearch/scripts/match_regions.py
source_extraction_autoresearch/scripts/normalize_text.py
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
10. the improvement did not come from editing benchmark/scoring/logging files.
```

Otherwise revert the experiment change and append a failed result row. Every benchmark run appends exactly one row to `results/results.tsv`.

