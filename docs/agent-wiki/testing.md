# Testing

## Start Here

The default test entrypoint is `.testing/run_tests.ps1`. Use targeted tests while iterating, then broader checks when behavior crosses subsystem boundaries.

## Main Commands

```powershell
.\.testing\run_tests.ps1
```

```powershell
.\.testing\run_tests.ps1 -WithSmoke
```

```powershell
python -m manga_local_translator --help
python -m manga_local_translator.quality_eval --help
```

Render-only focused check:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .testing\tests -p test_render.py
```

## Test Categories

| Area | Likely tests |
|---|---|
| CLI/help/import sanity | `.testing/run_tests.ps1` |
| Detection/OCR heuristics | `test_detect_ocr_heuristics.py`, `test_source_extraction.py`, `test_source_extraction_score.py` |
| Text filtering/rules | `test_text_filter.py`, `test_translation_rules.py`, `test_translation_evidence.py` |
| Grouping/identity/cache | `test_grouping.py`, `test_line_identity.py`, `test_page_cache.py` |
| Rendering/erase/debug | `test_render.py`, `test_erase.py`, `test_debug_report.py` |
| Qwen/CAT/vision | `test_qwen_*`, `test_cat_translator.py`, `test_vision_*` |
| Quality/review tooling | `test_quality_eval.py`, `test_review_report.py` |
| Autoresearch harnesses | `test_autoresearch_harness.py`, harness-specific tests |

## Smoke Data

Some smoke tests expect local images. `.testing/input/` and `mangafolder/` are ignored/local data in this checkout. Do not assume they exist in every clone.

## Quality Eval

Use quality eval for repeatable translation/report experiments:

```powershell
python -m manga_local_translator.quality_eval ".\raw_pages" `
  --output-root ".\quality-runs" `
  --name "my-test" `
  --profiles quality `
  --translator qwen `
  --qwen-mode page
```

By default, quality eval skips final image rendering. Add `--render-images` for rendered pages, or `--render-only --resume` to render from cache.

## When To Run What

| Change | Minimum useful check |
|---|---|
| Docs only | Link/path validation and CLI help if CLI docs changed. |
| CLI/config | `python -m manga_local_translator --help`, relevant config tests. |
| OCR/filtering | OCR/filter/source extraction tests. |
| Translation rules | translation rule/evidence/Qwen/CAT tests as applicable. |
| Render layout | `test_render.py`; render harness if optimizing quality. |
| Cache/schema | `test_page_cache.py`, debug/review/quality tests. |
| Cross-pipeline behavior | `.testing/run_tests.ps1`; optional smoke if data/models are available. |

