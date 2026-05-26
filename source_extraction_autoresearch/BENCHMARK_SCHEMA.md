# Benchmark Schema

Each benchmark set is a folder containing page images plus UTF-8 JSON labels:

```text
source_extraction_autoresearch/benchmarks/<benchmark_name>/
  README.md
  pages/
    page_001.png
  labels/
    page_001.labels.json
  ignore_regions/
    page_001.ignore.json
  configs/
    page_001.config.json
```

Real hand-labeled case sets live under `benchmarks/cases/<case_set_name>/`.

Label JSON required page-level fields:

```text
schema_version
page_id
image
text_regions
groups
```

Required `text_regions` fields:

```text
region_id
box
source_text
orientation
kind
should_extract
```

Extractable regions also require:

```text
group_id
page_order
```

Allowed orientations: `vertical`, `horizontal`, `mixed`, `unknown`.

Allowed kinds: `dialogue`, `narration`, `sfx`, `sign`, `thought`, `metadata`, `credit`, `page_number`, `noise`, `unknown`.

Normally extractable kinds: `dialogue`, `narration`, `thought`, `sign`, `sfx`.

Example label:

```json
{
  "schema_version": 1,
  "page_id": "page_001",
  "image": "pages/page_001.png",
  "image_size": [1200, 1800],
  "text_regions": [
    {
      "region_id": "r001",
      "box": [820, 120, 910, 420],
      "source_text": "おはよう",
      "normalized_source_text": "おはよう",
      "orientation": "vertical",
      "group_id": "g001",
      "page_order": 1,
      "kind": "dialogue",
      "should_extract": true,
      "difficulty_tags": ["vertical", "speech_bubble"]
    }
  ],
  "groups": [
    {
      "group_id": "g001",
      "member_region_ids": ["r001"],
      "combined_source_text": "おはよう",
      "page_order": 1,
      "kind": "dialogue"
    }
  ],
  "notes": "Simple vertical speech bubble case."
}
```

Ignore-region JSON:

```json
{
  "schema_version": 1,
  "page_id": "page_001",
  "ignore_regions": [
    {
      "box": [0, 0, 1200, 80],
      "reason": "scan header / crop margin"
    }
  ]
}
```

Run-config JSON:

```json
{
  "schema_version": 1,
  "page_id": "page_001",
  "detector": "ctd",
  "ocr_engine": "manga-ocr",
  "allow_tesseract_fallback": true,
  "notes": "Use normal project defaults unless this page needs a specific detector."
}
```

Predicted output schema returned by `scripts/io_adapters.py`:

```json
{
  "schema_version": 1,
  "page_id": "page_001",
  "image_path": "source_extraction_autoresearch/benchmarks/synthetic/pages/page_001.png",
  "adapter_mode": "project",
  "regions": [
    {
      "pred_region_id": "p001",
      "box": [818, 118, 912, 423],
      "ocr_text": "おはよう",
      "normalized_ocr_text": "おはよう",
      "confidence": 92.5,
      "orientation": "vertical",
      "kind": "dialogue",
      "group_id": "pg001",
      "page_order": 1,
      "source_engine": "manga-ocr",
      "detector": "ctd",
      "metadata": {}
    }
  ],
  "groups": [
    {
      "pred_group_id": "pg001",
      "member_pred_region_ids": ["p001"],
      "combined_ocr_text": "おはよう",
      "page_order": 1,
      "kind": "dialogue"
    }
  ],
  "timing": {
    "detect_ms": 0.0,
    "ocr_ms": 0.0,
    "group_ms": 0.0,
    "total_ms": 368.0
  },
  "adapter_notes": []
}
```

