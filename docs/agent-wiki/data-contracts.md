# Data Contracts

## Start Here

Use this page before changing dataclasses, cache files, debug report fields, or review outputs. Many tools consume these shapes indirectly.

## Core Types

| Type | Source | Purpose |
|---|---|---|
| `TextBlock` | `manga_local_translator/detect_types.py` | A text region with source text, box, confidence, detector, and metadata. |
| `PreparedPage` | `manga_local_translator/page_types.py` | Page-level state after detection/OCR/grouping and before or after translation. |
| `PipelineConfig` | `manga_local_translator/config.py` | Immutable runtime settings shared by entrypoints and pipeline stages. |
| `VisionArtifact` and related request/result types | `vision_types.py` | Number maps, vision repair requests, visual facts, and validated results. |

`TextBlock.box` is a tuple of four integers. Keep box semantics consistent across OCR, grouping, erase, render, debug, and cache code.

## PreparedPage Contents

`PreparedPage` carries:

- source and output paths,
- original page image data and dimensions,
- raw OCR blocks,
- grouped render blocks,
- skipped-block reports,
- grouping and page-order reports,
- translations keyed by line/block identity,
- translation contexts with debug and routing metadata,
- fallback block reports,
- optional vision artifacts.

`chapter_context_after` currently returns the final page-order source text when available.

## Line Identity

`line_identity.py` assigns stable-ish IDs and provides helper accessors:

- `assign_ocr_block_ids()`
- `assign_render_line_ids()`
- `translation_key_for_block()`
- `lookup_translation()` / `set_translation()`
- `lookup_context()` / `set_context()`
- cache compatibility helpers

Use these helpers instead of direct source-text keys when touching translation state.

## Cache Files

`page_cache.py` serializes prepared pages and translation stages. Cache file names are built from page index, image stem, and stage.

Common stages:

| Stage | Meaning |
|---|---|
| `prepared` | Detection/OCR/grouping data before translation. |
| `primary...` | Primary translator outputs. |
| `critic...` | Primary outputs with critic metadata. |
| `hybrid...` | Final hybrid/fallback outputs. |

Stage names can include Qwen mode, CAT prompt/cache identifiers, critic/fallback state, and vision facts. Render-only lookup accepts a small set of expected final stages rather than every historical cache shape.

Abbreviated cache examples:

```text
.manga-work/
  0001_page-001.prepared.json
  0001_page-001.primary-opus.json
  0001_page-001.hybrid-qwen-page-critic-fallback-vision-facts.json
```

```json
{
  "source_path": "raw_pages/page-001.png",
  "image_size": [1200, 1800],
  "raw_blocks": [
    {
      "line_id": "ocr-0001",
      "text": "日本語",
      "box": [100, 120, 220, 180],
      "confidence": 92.5,
      "detector": "ctd"
    }
  ],
  "render_blocks": [
    {
      "line_id": "line-0001",
      "source_text": "日本語",
      "box": [96, 116, 228, 188]
    }
  ],
  "translations": {
    "line-0001": "Japanese"
  }
}
```

## Debug Reports

`debug_report.py` writes `.ocr.json` reports. These are used by humans, `review_report.py`, and `quality_eval.py`.

Important report areas:

- kept blocks,
- skipped blocks,
- fallback blocks,
- source and translated text,
- detector/OCR metadata,
- line IDs and page order,
- render boxes/fits/warnings,
- Qwen/CAT/critic/fallback metadata,
- vision and vision-facts summaries,
- page-level summary counts.

If a field is renamed or removed, inspect `review_report.py`, `quality_eval.py`, and tests before changing it.

Abbreviated `.ocr.json` shape:

```json
{
  "source": "raw_pages/page-001.png",
  "output": "translated/page-001.png",
  "summary": {
    "kept_blocks": 1,
    "skipped_blocks": 0,
    "fallback_blocks": 0,
    "warnings": 0
  },
  "blocks": [
    {
      "line_id": "line-0001",
      "source_text": "日本語",
      "translated_text": "Japanese",
      "box": [100, 120, 220, 180],
      "render_box": [96, 116, 228, 188],
      "confidence": 92.5,
      "detector": "ctd",
      "translation_context": {
        "translator": "opus",
        "rejected": false,
        "fallback_used": false
      },
      "render": {
        "font_size": 28,
        "fit": "ok",
        "warnings": []
      }
    }
  ],
  "skipped_blocks": [],
  "page_order": ["line-0001"],
  "vision": null
}
```

The examples show representative field groupings, not an exhaustive schema. Prefer additive changes when extending debug or cache data.

## Review And Quality Outputs

`review_report.py` converts `.ocr.json` files into CSV, HTML, and Markdown review tables.

`quality_eval.py` runs repeatable profile commands and summarizes debug reports into:

- `quality_summary.json`
- `quality_summary.csv`
- `translation_review.html`
- `translation_review.csv`
- `translation_review.md`

These outputs are usually generated under `quality-runs/` or `.testing/output/` and are not the canonical schema definition.

## Vision Artifacts

Vision support has two related but distinct paths:

- repair: asks vision to support translation repair for selected lines,
- facts: asks vision for visual context before translation.

`numbered_page` mode creates a masked/labeled artifact that maps visible line numbers to text blocks. This is safer than asking the vision model to read original manga text directly.
