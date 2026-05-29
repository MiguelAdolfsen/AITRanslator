# Debugging

## Start Here

For bad output, rerun with `--debug` and inspect the `.debug.png` and `.ocr.json` before tuning translation. Many translation problems are caused by OCR, grouping, or render constraints.

## Main Diagnostics

| Artifact | Source | Use |
|---|---|---|
| `manga_translator_debug.log` | `logging_utils.py` | Run-level logs, model setup, stage decisions. |
| `*.debug.png` | `debug_report.py` | Visual overlay of boxes, skipped blocks, and render diagnostics. |
| `*.ocr.json` | `debug_report.py` | OCR text, translations, contexts, warnings, and page summary. |
| `*.vision.png` / `*.vision-facts.png` | `vision_artifact.py` | What the vision model saw. |
| `.manga-work/` or configured work dir | `page_cache.py` | Prepared and translated cache stages for resume/render-only. |
| review CSV/HTML/Markdown | `review_report.py`, `quality_eval.py` | Human triage of many pages/blocks. |

## Fast Diagnosis Flow

1. Run with `--translator none --debug --overwrite` to isolate detection, OCR, grouping, erase, and render.
2. Inspect `.debug.png` for missing boxes, bad grouping, or render collisions.
3. Inspect `.ocr.json` for skipped blocks, source text quality, reading order, translation contexts, and layout warnings.
4. Only tune translation after source text and grouping look plausible.
5. For Qwen/CAT runs, inspect per-block context flags such as rejected, retry, critic, fallback, evidence, and vision fields.

## Common Problems

| Symptom | First places to inspect |
|---|---|
| No images processed | input path, supported extensions in `image_io.py`, recursive flag. |
| CTD unavailable | `ctd_setup.py`, `install_ctd.py`, README setup. |
| Tesseract unavailable | `tesseract_utils.py`, `--tesseract-cmd`, language packs. |
| Missing text | detector choice, `.debug.png`, `detect_ocr.py`, source extraction tests. |
| Bad Japanese source | OCR crop heuristics in `detect_ocr.py`, `text_filter.py`, `.ocr.json`. |
| Wrong reading order | `grouping.py`, `line_identity.py`, page-order report. |
| Bad translation | `translation_rules.py`, translator backend, Qwen/CAT debug fields, glossary. |
| Hallucinated repair | `translation_evidence.py`, `qwen_validation.py`, `vision_validation.py`. |
| Text does not fit | `render.py`, render warnings, `--font-size`, `--render-expand`. |
| Render-only fails | work dir, matching cache stages, original translator/model flags. |

## Useful Commands

```powershell
python -m manga_local_translator ".\raw_pages" ".\debug_output" `
  --translator none `
  --debug `
  --overwrite
```

```powershell
python -m manga_local_translator.quality_eval ".\raw_pages" `
  --output-root ".\quality-runs" `
  --name "debug-summary" `
  --summarize-only
```

```powershell
python -m manga_local_translator.review_report ".\debug_output"
```

## Debug Schema Care

Debug reports are consumed by review and quality tooling. When adding fields, prefer additive changes. When renaming or removing fields, update tests and [data-contracts.md](data-contracts.md).

