# Architecture

## Start Here

Use this page to find the right subsystem before editing. The orchestration center is `manga_local_translator/pipeline.py`; most other modules are stage implementations or support contracts.

## Subsystem Map

| Subsystem | Purpose | Source anchors |
|---|---|---|
| Entrypoints | Parse user input and start a run. | `manga_local_translator/cli.py`, `manga_local_translator/gui.py`, `manga_local_translator/__main__.py` |
| Configuration | Single runtime settings object shared by CLI, GUI, and pipeline. | `manga_local_translator/config.py` |
| Image I/O | Discover supported images, sort naturally, resolve output paths, write files. | `manga_local_translator/image_io.py` |
| Detection/OCR | Find Japanese text boxes and recognize text. | `manga_local_translator/detect_ocr.py`, `manga_local_translator/ctd_detector.py`, `manga_local_translator/source_extraction.py` |
| Filtering/grouping | Remove bad OCR fragments, group lines into render/translation blocks, assign reading order. | `text_filter.py`, `grouping.py`, `line_identity.py` |
| Translation | Build selected translator and postprocess outputs. | `translate.py`, `translator_base.py`, `translation_rules.py`, `hf_translators.py`, `argos_translator.py`, `qwen_translator.py` |
| Qwen validation/evidence | Parse Qwen output, validate translations, apply critic/fallback decisions. | `qwen_validation.py`, `qwen_types.py`, `translation_evidence.py`, `translation_review.py`, `translation_review_pass.py`, `pipeline.py` |
| Vision | Create numbered artifacts, prompt vision model, validate visual facts/repairs. | `vision_artifact.py`, `vision_prompt.py`, `vision_service.py`, `vision_validation.py`, `vision_types.py`, `qwen_vision.py` |
| Erase/render | Remove original text and typeset English translation. | `erase.py`, `render.py` |
| Debug/review | Write overlays, JSON reports, review CSV/HTML/Markdown. | `debug_report.py`, `review_report.py`, `quality_eval.py` |
| Cache/resume | Save prepared pages and translation stages for resume/render-only flows. | `cache_policy.py`, `page_cache.py`, `page_types.py` |
| Model installers | Download or prepare optional model assets. | `install_ctd.py`, `install_opus.py`, `install_hy_mt2.py`, `install_madlad.py`, `install_argos.py`, `install_cat.py` |

## Main Runtime Ownership

`process_folder()` in `pipeline.py` owns run-level orchestration:

1. Resolve input/output paths.
2. Enable offline Hugging Face runtime behavior during processing.
3. Validate detector prerequisites.
4. Discover images through `iter_images()`.
5. Choose standard per-page flow, batched Qwen/CAT hybrid flow, or render-only cache flow.
6. Build translators only after prerequisites and image discovery.
7. Restore environment variables at the end.

Per-page state is represented as `PreparedPage`. Individual source or render regions are represented as `TextBlock`.

## Where To Change Behavior

| If you need to change... | Edit first | Also inspect |
|---|---|---|
| Accepted CLI flags or defaults | `cli.py`, `config.py` | `gui.py`, README, [configuration](configuration.md) |
| GUI-only defaults or controls | `gui.py` | `config.py`, `cli.py` for parity |
| OCR candidate generation | `detect_ocr.py`, `ctd_detector.py` | `source_extraction.py`, tests |
| OCR skip rules | `text_filter.py` | `debug_report.py` so reports explain skips |
| Reading order/grouping | `grouping.py`, `line_identity.py` | cache and debug report tests |
| Translation cleanup | `translation_rules.py` | `translation_evidence.py`, `translation_review.py`, `translation_review_pass.py`, translator tests |
| CAT behavior | `hf_translators.py` | `cat_response_autoresearch/`, CAT tests |
| Qwen prompts/parsing | `qwen_prompts.py`, `qwen_validation.py`, `qwen_translator.py` | Qwen tests and quality eval |
| Vision repair/facts | `vision_*`, `qwen_vision.py` | vision tests and debug report fields |
| Render fitting | `render.py` | `render_autoresearch/PROGRAM.md` before benchmarking |
| Debug schema | `debug_report.py` | `review_report.py`, `quality_eval.py`, [data contracts](data-contracts.md) |
