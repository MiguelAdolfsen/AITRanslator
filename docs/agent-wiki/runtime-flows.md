# Runtime Flows

## Start Here

Use this page when changing orchestration. Most flow decisions are in `pipeline.py`, with settings from `PipelineConfig`.

## Standard CLI Flow

The CLI path is:

```text
cli.build_parser()
  -> PipelineConfig(...)
  -> pipeline.process_folder(input, output, config)
  -> build_translator(...)
  -> prepare_page_for_translation(...) for each page
  -> translate_prepared_page(...)
  -> render_prepared_page(...)
  -> render/debug outputs
```

This flow is used when the run does not need batched hybrid handling and is not `--render-only`.

## GUI Flow

`gui.py` builds a Tkinter form, converts widget values into `PipelineConfig`, checks runtime dependencies, then calls `process_folder()` in a worker thread. GUI install buttons call the same installer modules used from the command line.

Keep GUI and CLI behavior aligned when adding options unless the option is intentionally CLI-only.

## Hybrid Batch Flow

`process_folder_qwen_hybrid_batch()` is selected when:

- `translator == "cat"`, or
- `translator` is `qwen` or `hy-mt2`, and a fallback model, critic model, or pre-translation vision facts are enabled.

The batch flow prepares all pages first, then runs translation stages:

```text
prepare page cache
  -> optional vision facts
  -> primary translator pass
  -> optional CAT retry
  -> optional evidence memory
  -> optional Qwen critic
  -> optional Qwen fallback
  -> refresh fallback blocks
  -> render or write debug reports
```

Cache stages are named by `translation_cache_stage()`. The exact stage suffix can include translator, Qwen mode, CAT prompt versions, critic/fallback state, and vision facts.
Translation review pass behavior lives in `translation_review.py`; `pipeline.py` decides when each pass runs.

## Render-Only Flow

`--render-only` skips detection, OCR, and translation. It requires an existing work directory with:

- a prepared page cache for each input page, and
- a matching translation cache stage.

`render_cached_pages_only()` loads cached `PreparedPage` data, applies translations, refreshes fallback blocks, and renders images. It fails loudly if cache files are incomplete or flags do not match the original run.

## Resume Behavior

`--resume` reuses prepared and translation cache files when they exist. It is most important for Qwen/CAT hybrid runs because model inference is expensive.

If cache behavior changes, update:

- `page_cache.py`
- `pipeline.py`
- [data-contracts.md](data-contracts.md)
- tests in `.testing/tests/test_page_cache.py` and any affected pipeline tests

## Outputs

| Output | Created when | Purpose |
|---|---|---|
| translated image | default unless `--skip-render` | Final rendered page. |
| `*.debug.png` | `--debug` | Overlay of detected/rendered regions. |
| `*.ocr.json` | `--debug` or debug/report paths | OCR, translation, context, warnings, and summary fields. |
| `*.vision.png` / `*.vision-facts.png` | vision paths | Artifacts sent to vision model. |
| work cache JSON | `--resume`, hybrid, render-only workflows | Prepared and translated page state. |
| review reports | quality eval/review tools | CSV, HTML, Markdown inspection reports. |
