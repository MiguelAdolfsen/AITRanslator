# Codex setup spec: isolated source-extraction autoresearch loop for AITRanslator

You are Codex working inside the `AITRanslator` repository root. Your task is to set up an isolated autoresearch loop for the next optimization target after rendering and translation routing: **source extraction**.

Source extraction means this part of the pipeline only:

```text
page image -> text detection -> crop/OCR -> bad-fragment filtering -> grouping -> page/reading order
```

The loop must be separate from the main project. Do **not** put loop scripts, benchmark cases, run outputs, result logs, generated reports, or helper scripts into `.testing/`, `manga_local_translator/`, `quality-runs/`, `render_autoresearch/`, `translation_routing_autoresearch/`, or any other existing project folder. Everything for this loop must live under one new root-level folder:

```text
source_extraction_autoresearch/
```

The main project is only an import target and, later, the code under test. The autoresearch harness itself is a separate research system.

---

## 1. Non-negotiable separation rule

Create this folder in the repository root:

```text
source_extraction_autoresearch/
```

All new loop assets must go inside it.

Allowed new files during setup:

```text
source_extraction_autoresearch/**
```

Forbidden setup locations:

```text
.testing/**
manga_local_translator/**
quality-runs/**
.models/**
.model*/**
render_autoresearch/**
translation_routing_autoresearch/**
root-level helper scripts such as eval_source_extraction.py
root-level result files such as source_extraction_results.tsv
root-level generated benchmark folders
```

The setup phase must not edit main project source code. It may only read project files and create the isolated loop folder.

After setup, future optimization experiments may edit a very small project-code surface, but benchmark fixtures, scoring code, result logs, and run artifacts must remain inside `source_extraction_autoresearch/`.

If this handoff file is present at repository root as `codex_source_extraction_autoresearch_setup.md`, treat it only as a temporary instruction file. Copy its contents into:

```text
source_extraction_autoresearch/CODEX_SETUP.md
```

Do not create any other root-level setup files.

---

## 2. Purpose of the loop

The loop should optimize **Japanese source extraction quality** before translation or rendering happens.

The benchmark should evaluate whether the project can produce the correct source blocks from a page image:

```text
- detect the right text regions,
- avoid non-dialogue false positives,
- crop regions well enough for OCR,
- OCR Japanese text accurately,
- classify orientation correctly,
- group related text regions into translation units,
- keep separate bubbles or SFX separate,
- preserve stable line/block IDs,
- and produce correct manga reading order.
```

Lower score is better.

---

## 3. What this loop is not

Do not optimize or benchmark:

```text
English translation quality
translation routing
CAT/Qwen/OPUS/MADLAD/Argos behavior
Qwen vision repair
rendering/layout/font fitting
erasing/inpainting/art restoration
GUI behavior
model download/setup behavior
```

Rendering and translation routing have separate loops. Do not use their folders, fixtures, score files, or result logs for this source-extraction loop.

The evaluator must not call translation, rendering, Qwen, CAT, OPUS, MADLAD, Argos, Ollama, llama.cpp, or vision repair. It may call detection/OCR/grouping code only.

---

## 4. Project files Codex should inspect before implementation

Read these files to understand the current pipeline, debug outputs, and available helper functions:

```text
README.md
manga_local_translator/pipeline.py
manga_local_translator/config.py
manga_local_translator/cli.py
manga_local_translator/detect_ocr.py
manga_local_translator/detect_types.py
manga_local_translator/ctd_detector.py
manga_local_translator/grouping.py
manga_local_translator/text_filter.py
manga_local_translator/debug_report.py
manga_local_translator/line_identity.py
manga_local_translator/quality_eval.py
.testing/tests/
```

If a listed file does not exist in the local checkout, do not create it during setup. Continue with the files that exist.

Do not copy large chunks of main-project logic into the harness. Import and call project functions directly where useful.

For this repository, the preferred project-mode adapter path is:

```text
manga_local_translator.pipeline.prepare_page_for_translation(...)
```

That function currently performs the source-extraction portion only:

```text
image read -> run_ocr -> assign_ocr_block_ids -> filter_text_blocks -> group_text_blocks_for_translation -> build/enrich page order
```

Map its returned `PreparedPage` fields into the predicted-output schema:

```text
raw_blocks        -> raw detected/OCR regions
render_blocks     -> grouped source units
skipped_blocks    -> filtered/rejected source regions
grouping_report   -> group membership diagnostics
page_order_report -> reading order/context diagnostics
```

Do not use `process_image(...)` for this loop; it imports translation, erase, render, and vision stages.

The harness should tolerate minor project signature changes. Put fragile imports and project-specific glue behind one small adapter layer:

```text
source_extraction_autoresearch/scripts/io_adapters.py
```

---

## 5. Required folder tree

Create this structure exactly:

```text
source_extraction_autoresearch/
  README.md
  CODEX_SETUP.md
  PROGRAM.md
  BENCHMARK_SCHEMA.md
  SCORING.md
  .gitignore

  benchmarks/
    README.md
    cases/
      .gitkeep
    synthetic/
      .gitkeep
    fixtures/
      .gitkeep

  results/
    results.tsv
    best.json
    .gitkeep

  runs/
    .gitkeep

  scripts/
    __init__.py
    build_case_from_debug.py
    compare_runs.py
    eval_source_extraction.py
    generate_synthetic_cases.py
    io_adapters.py
    match_regions.py
    normalize_text.py
    score.py
    summarize_results.py
    validate_fixtures.py
    visualize_overlays.py

  templates/
    label.example.json
    ignore_regions.example.json
    run_config.example.json
    result_header.tsv
```

Put this setup spec itself inside the loop folder as:

```text
source_extraction_autoresearch/CODEX_SETUP.md
```

No setup files should be placed outside `source_extraction_autoresearch/`.

---

## 6. File responsibilities

### `source_extraction_autoresearch/README.md`

Explain the loop in human terms:

- It is isolated from the main project.
- It benchmarks only detection, OCR, filtering, grouping, orientation, and reading order.
- It uses frozen page images and hand-labeled ground-truth source-region files.
- It does not run translation, routing, Qwen, CAT, OPUS, rendering, erase, inpainting, or GUI code.
- It imports the main project as a library.
- It writes append-only result logs.
- It is intended to help future autoresearch agents decide which small source-code changes to keep or revert.

Include quick commands:

```bash
python source_extraction_autoresearch/scripts/generate_synthetic_cases.py \
  --output source_extraction_autoresearch/benchmarks/synthetic

python source_extraction_autoresearch/scripts/validate_fixtures.py \
  --benchmark source_extraction_autoresearch/benchmarks/synthetic

python source_extraction_autoresearch/scripts/eval_source_extraction.py \
  --benchmark source_extraction_autoresearch/benchmarks/synthetic \
  --output source_extraction_autoresearch/runs/baseline \
  --results source_extraction_autoresearch/results/results.tsv \
  --run-id baseline

python source_extraction_autoresearch/scripts/summarize_results.py \
  --results source_extraction_autoresearch/results/results.tsv
```

On Windows in this project, prefer the repo virtualenv when present:

```powershell
.\.venv\Scripts\python.exe source_extraction_autoresearch\scripts\generate_synthetic_cases.py `
  --output source_extraction_autoresearch\benchmarks\synthetic
```

### `source_extraction_autoresearch/CODEX_SETUP.md`

Copy this full setup spec into that file.

### `source_extraction_autoresearch/PROGRAM.md`

This is the agent-facing autoresearch instruction file. It should tell a future coding agent how to run experiments.

It must include:

- Objective: minimize `source_extraction_score`.
- Scope: detection, OCR crop preparation, OCR acceptance, text filtering, grouping, orientation, and reading order only.
- Frozen benchmark rule: do not modify benchmark cases, labels, scoring, templates, or result-log schema during optimization.
- Required setup command.
- Required fixture-validation command.
- Required unit-test command.
- Required benchmark command.
- Editable files by phase.
- Forbidden edits.
- Keep/revert rules.
- Result-log requirements.

Use this objective statement:

```text
Minimize missed, false, garbled, wrongly grouped, and wrongly ordered Japanese source blocks before translation.
```

Use this phase-1 editable project-code surface:

```text
manga_local_translator/detect_ocr.py
manga_local_translator/grouping.py
manga_local_translator/text_filter.py
```

Use this phase-2 editable surface only after the phase-1 loop is stable:

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

Required commands before accepting an experiment:

```powershell
.\.testing\run_tests.ps1

.\.venv\Scripts\python.exe source_extraction_autoresearch\scripts\validate_fixtures.py `
  --benchmark source_extraction_autoresearch\benchmarks\synthetic

.\.venv\Scripts\python.exe source_extraction_autoresearch\scripts\eval_source_extraction.py `
  --benchmark source_extraction_autoresearch\benchmarks\synthetic `
  --output source_extraction_autoresearch\runs\current `
  --results source_extraction_autoresearch\results\results.tsv `
  --run-id current `
  --tests-ok
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

Otherwise revert the change and append a failed result row.

### `source_extraction_autoresearch/BENCHMARK_SCHEMA.md`

Define the fixture schema exactly. See section 8 below.

### `source_extraction_autoresearch/SCORING.md`

Define the scoring formula exactly. See section 11 below.

### `source_extraction_autoresearch/.gitignore`

Keep generated run noise out of commits, but keep schemas, scripts, labels, benchmark definitions, and result logs visible.

Recommended content:

```gitignore
runs/*
!runs/.gitkeep
results/*.tmp
results/*.log
results/*.jsonl
!results/.gitkeep
!results/results.tsv
!results/best.json
__pycache__/
*.pyc
.DS_Store
```

### `source_extraction_autoresearch/results/results.tsv`

Create this file with the header from `templates/result_header.tsv`.

The log is append-only.

```text
Never delete previous rows.
Never silently rewrite previous rows.
Append exactly one row per benchmark run.
```

### `source_extraction_autoresearch/results/best.json`

Create a small JSON file tracking the current best run:

```json
{
  "schema_version": 1,
  "best_run_id": null,
  "best_score": null,
  "best_commit": null,
  "updated_at": null,
  "notes": "Updated by eval_source_extraction.py only when a run improves the score and passes hard guards."
}
```

---

## 7. Benchmarking concept

The benchmark should evaluate **source extraction over frozen page images and hand-labeled ground truth**.

For each page, the evaluator loads:

```text
image file
label file
ignore-region file, if present
run config, if present
```

Then it calls the project source-extraction path through `io_adapters.py` and collects predicted outputs:

```text
predicted text regions
predicted OCR text
predicted orientation
predicted grouping
predicted reading order
source-extraction timing
```

The evaluator compares predictions with labels, writes per-run reports, computes one lower-is-better score, and appends one row to the result log.

The benchmark should support two modes:

```text
project mode:
  Import current project detection/OCR/grouping code through io_adapters.py.
  This is the real optimization mode.

adapter-smoke mode:
  Use a deterministic fake predictor only to verify that fixture validation,
  matching, scoring, result logs, and output writing work.
  This mode is for harness setup only and must not be used to claim project improvements.
```

The evaluator must never call translation or rendering.

If required OCR/detector dependencies are missing, the evaluator should fail clearly unless `--adapter-smoke` is explicitly passed.

---

## 8. Benchmark fixture schema

Each benchmark set is a folder containing page images plus JSON labels:

```text
source_extraction_autoresearch/benchmarks/<benchmark_name>/
  README.md
  pages/
    page_001.png
    page_002.png
  labels/
    page_001.labels.json
    page_002.labels.json
  ignore_regions/
    page_001.ignore.json       # optional
    page_002.ignore.json       # optional
  configs/
    page_001.config.json       # optional
    page_002.config.json       # optional
```

For synthetic setup, create:

```text
source_extraction_autoresearch/benchmarks/synthetic/
  README.md
  pages/
  labels/
  ignore_regions/
  configs/
```

For real hand-labeled pages, create:

```text
source_extraction_autoresearch/benchmarks/cases/<case_set_name>/
  README.md
  pages/
  labels/
  ignore_regions/
  configs/
```

All JSON files containing Japanese text must be read and written as UTF-8.

In Python, use:

```python
path.read_text(encoding="utf-8")
json.dumps(obj, ensure_ascii=False, indent=2)
```

### `labels/<page_id>.labels.json`

Example:

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
    },
    {
      "region_id": "r002",
      "box": [100, 1600, 180, 1640],
      "source_text": "第3話",
      "normalized_source_text": "第3話",
      "orientation": "horizontal",
      "group_id": null,
      "page_order": null,
      "kind": "metadata",
      "should_extract": false,
      "difficulty_tags": ["chapter_label"]
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

Required page-level fields:

```text
schema_version
page_id
image
text_regions
groups
```

Required fields for every `text_regions` entry:

```text
region_id
box
source_text
orientation
kind
should_extract
```

Required fields for extractable regions where `should_extract` is `true`:

```text
group_id
page_order
```

Allowed `orientation` values:

```text
vertical
horizontal
mixed
unknown
```

Allowed `kind` values:

```text
dialogue
narration
sfx
sign
thought
metadata
credit
page_number
noise
unknown
```

Only these kinds are normally expected to be translated:

```text
dialogue
narration
thought
sign
sfx
```

Metadata, credits, page numbers, and noise are usually `should_extract: false` unless a benchmark case explicitly marks them extractable.

### `ignore_regions/<page_id>.ignore.json`

Optional file for areas that should not count as false positives.

Example:

```json
{
  "schema_version": 1,
  "page_id": "page_001",
  "ignore_regions": [
    {
      "box": [0, 0, 1200, 80],
      "reason": "scan header / crop margin"
    },
    {
      "box": [40, 1720, 260, 1780],
      "reason": "publisher logo"
    }
  ]
}
```

Ignore regions are only for known non-content zones. Do not use ignore regions to hide real extraction mistakes.

### `configs/<page_id>.config.json`

Optional per-page run configuration.

Example:

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

If no config exists, the evaluator should use a benchmark-level default config.

---

## 9. Predicted-output schema

`io_adapters.py` should expose one narrow interface used by the evaluator:

```python
def extract_source_page(image_path: str, page_config: dict | None = None) -> dict:
    """Return predicted source-extraction outputs for one page image."""
```

The returned dictionary must have this shape:

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
    "detect_ms": 120.0,
    "ocr_ms": 240.0,
    "group_ms": 8.0,
    "total_ms": 368.0
  },
  "adapter_notes": []
}
```

Required fields for predicted regions:

```text
pred_region_id
box
ocr_text
orientation
```

Recommended fields:

```text
normalized_ocr_text
confidence
kind
group_id
page_order
source_engine
detector
metadata
```

Adapter rules:

```text
Prefer importing existing project helpers.
Prefer `manga_local_translator.pipeline.prepare_page_for_translation(...)` for project mode.
Do not use `manga_local_translator.pipeline.process_image(...)`.
Do not call translation, rendering, erase, Qwen, CAT, OPUS, MADLAD, Argos, vision, or GUI code.
Do not call network services.
Do not download models.
If project helpers are unavailable or signatures changed, fail clearly unless --adapter-smoke is explicitly used.
Write adapter notes into run config and summary output.
```

---

## 10. Matching and normalization

### Text normalization

Create `scripts/normalize_text.py` with deterministic helpers.

Use Unicode NFKC normalization and remove insignificant whitespace for Japanese OCR comparison. Keep punctuation meaningful enough to catch OCR losses.

Recommended normalization steps:

```text
1. Unicode normalize with NFKC.
2. Strip leading/trailing whitespace.
3. Collapse internal whitespace.
4. Normalize common ellipsis forms to `…`.
5. Normalize repeated full-width spaces.
6. Preserve kana, kanji, Latin letters, digits, punctuation, and bracket/code-like tokens.
```

Do not aggressively rewrite Japanese. The goal is to compare OCR output, not to hide errors.

### Character error rate

Implement a pure-Python Levenshtein distance in `normalize_text.py` or `score.py`.

```text
cer = levenshtein(normalized_prediction, normalized_reference) / max(1, len(normalized_reference))
```

Cap CER at `1.0` for aggregate scoring.

### Region matching

Create `scripts/match_regions.py`.

Match predicted regions to ground-truth extractable regions using box IoU first, then text similarity.

Recommended match score:

```text
match_score = 0.65 * box_iou + 0.35 * text_similarity
```

Where:

```text
text_similarity = 1 - capped_cer
```

A predicted region is a valid match when either:

```text
box_iou >= 0.45
```

or, for very small regions:

```text
predicted_box_center_inside_ground_truth_box
and text_similarity >= 0.60
```

Use one-to-one matching. One predicted region cannot match multiple ground-truth regions. One ground-truth region cannot match multiple predictions.

### Duplicate detection

A duplicate prediction is an unmatched predicted region that has high overlap with an already matched predicted region or another unmatched prediction:

```text
IoU >= 0.70
and normalized OCR text is identical or highly similar
```

### Destructive false positives

A false positive is destructive when it would likely cause fake translation or unwanted erase/render later. Count it as destructive if it:

```text
- does not match any extractable ground-truth region,
- is not mostly inside an ignore region,
- and overlaps a labeled non-extractable region such as metadata, credit, page number, or noise,
```

or if it has plausible OCR text but is on background/art with no labeled source region.

Harmless false positives are low-confidence or tiny regions mostly inside ignore regions or explicit non-content margins.

### Group matching

Use matched region IDs to compare predicted groups to labeled groups.

Count an overmerge when:

```text
one predicted group contains matched regions from multiple ground-truth group_id values
```

Count an undermerge when:

```text
one ground-truth group_id is split across multiple predicted groups
```

### Reading order

Compare the page order of matched extractable regions or groups.

Use pairwise inversions:

```text
reading_order_error_rate = inverted_matched_pairs / max(1, comparable_matched_pairs)
```

Do not punish unmatched regions twice. Missed regions are already counted by region recall; reading order should be calculated over matched extractable regions/groups only.

---

## 11. Scoring

Create `source_extraction_autoresearch/SCORING.md` and `source_extraction_autoresearch/scripts/score.py` with this formula.

Primary metric:

```text
source_extraction_score
```

Lower is better.

Formula:

```text
source_extraction_score =
  12000 * missed_dialogue_region_rate
+ 10000 * destructive_false_positive_rate
+  8000 * wrong_text_region_match_rate
+  7000 * mean_ocr_cer
+  6500 * severe_ocr_error_rate
+  6000 * overmerge_rate
+  6000 * undermerge_rate
+  5000 * reading_order_error_rate
+  4000 * orientation_error_rate
+  3000 * bad_crop_rate
+  2500 * duplicate_region_rate
+  1800 * empty_ocr_rate
+  1500 * non_japanese_noise_rate
+   500 * harmless_false_positive_rate
+     1 * p95_extraction_ms_per_page
```

Definitions:

```text
missed_dialogue_region_rate:
  Extractable dialogue/narration/thought/sign/SFX ground-truth regions with no matching prediction, divided by extractable ground-truth regions.

destructive_false_positive_rate:
  Predicted regions that would likely create fake translation or unwanted erase/render, divided by predicted regions.

wrong_text_region_match_rate:
  Matched regions whose OCR text is severely inconsistent with reference text even though the box matched.

mean_ocr_cer:
  Mean capped character error rate over matched extractable regions.

severe_ocr_error_rate:
  Matched extractable regions with capped CER >= 0.50.

overmerge_rate:
  Predicted groups that combine multiple distinct ground-truth groups, divided by predicted groups.

undermerge_rate:
  Ground-truth groups split across multiple predicted groups, divided by ground-truth groups.

reading_order_error_rate:
  Pairwise order inversions over matched comparable groups or regions.

orientation_error_rate:
  Matched extractable regions whose predicted orientation differs from the ground-truth orientation.

bad_crop_rate:
  Matched regions with weak spatial fit, such as IoU < 0.45 and no small-region exception, or boxes that cut off visible glyph area according to labels.

duplicate_region_rate:
  Duplicate predicted regions divided by predicted regions.

empty_ocr_rate:
  Matched extractable regions with empty or punctuation-only OCR output.

non_japanese_noise_rate:
  Predicted regions accepted as source text but dominated by non-Japanese noise when the label says they are not extractable.

harmless_false_positive_rate:
  Low-impact false positives, divided by predicted regions.

p95_extraction_ms_per_page:
  95th percentile total source-extraction time per page.
```

Hard failure conditions:

```text
unit tests fail
fixture validation fails
evaluator crashes
results row is not appended
benchmark/scoring/matching/normalization files were edited during an optimization run
any benchmark label file is edited during an optimization run
all OCR text is blanked or replaced with placeholders
predicted region count becomes zero on pages with extractable labels
line/block IDs become unstable for identical input across two runs
invalid JSON in run outputs
NaN, infinite, or missing source_extraction_score
translation, rendering, Qwen, CAT, OPUS, MADLAD, Argos, or vision code is invoked by the evaluator
```

Tie-breakers, in order:

```text
1. lower missed_dialogue_region_count
2. lower destructive_false_positive_count
3. lower mean_ocr_cer
4. lower severe_ocr_error_count
5. lower overmerge_count + undermerge_count
6. lower reading_order_error_count
7. lower orientation_error_count
8. lower duplicate_region_count
9. lower p95_extraction_ms_per_page
10. smaller project-code diff
```

---

## 12. Result log schema

Create:

```text
source_extraction_autoresearch/templates/result_header.tsv
source_extraction_autoresearch/results/results.tsv
```

Both should use exactly this header:

```text
run_id	timestamp	commit	parent_commit	experiment_name	changed_files	tests_ok	fixture_validation_ok	benchmark_set	source_extraction_score	baseline_score	delta_score	pages_total	gt_regions_total	gt_extractable_regions	predicted_regions	matched_regions	missed_dialogue_region_count	false_positive_region_count	destructive_false_positive_count	harmless_false_positive_count	mean_box_iou	mean_ocr_cer	severe_ocr_error_count	wrong_text_region_match_count	empty_ocr_count	overmerge_count	undermerge_count	reading_order_error_count	orientation_error_count	bad_crop_count	duplicate_region_count	non_japanese_noise_count	mean_extraction_ms_per_page	p95_extraction_ms_per_page	kept	notes
```

Rules:

```text
Append exactly one row per benchmark run.
Never delete previous rows.
Never silently rewrite previous rows.
Use ISO-8601 timestamps.
Use current git commit hash if available; otherwise write UNKNOWN.
Use semicolon-separated paths in changed_files.
Use TRUE/FALSE for booleans.
Use numeric zero instead of blanks for counts.
Use benchmark_set to record synthetic, cases/<name>, or another explicit benchmark name.
```

---

## 13. Run output schema

Each run writes to:

```text
source_extraction_autoresearch/runs/<run_id>/
```

Required files:

```text
summary.json
config.json
per_page_metrics.jsonl
region_matches.jsonl
predicted_regions.jsonl
predicted_groups.jsonl
failures.jsonl
overlays/
```

### `summary.json`

Example:

```json
{
  "schema_version": 1,
  "run_id": "baseline",
  "benchmark_set": "synthetic",
  "pages_total": 12,
  "source_extraction_score": 1234.56,
  "hard_failure": false,
  "metrics": {
    "gt_extractable_regions": 64,
    "predicted_regions": 66,
    "matched_regions": 60,
    "missed_dialogue_region_count": 4,
    "destructive_false_positive_count": 1,
    "mean_ocr_cer": 0.12,
    "overmerge_count": 2,
    "undermerge_count": 1,
    "reading_order_error_count": 3,
    "p95_extraction_ms_per_page": 900.0
  }
}
```

### `per_page_metrics.jsonl`

Each line should include:

```json
{
  "page_id": "page_001",
  "gt_extractable_regions": 5,
  "predicted_regions": 6,
  "matched_regions": 5,
  "missed_regions": 0,
  "false_positives": 1,
  "mean_ocr_cer": 0.08,
  "overmerge_count": 0,
  "undermerge_count": 0,
  "reading_order_error_count": 0,
  "orientation_error_count": 0,
  "total_ms": 512.0
}
```

### `region_matches.jsonl`

Each line should include one ground-truth/prediction comparison:

```json
{
  "page_id": "page_001",
  "region_id": "r001",
  "pred_region_id": "p001",
  "matched": true,
  "box_iou": 0.82,
  "text_similarity": 0.95,
  "cer": 0.05,
  "orientation_ok": true,
  "gt_text": "おはよう",
  "pred_text": "おはよう",
  "violations": []
}
```

### `predicted_regions.jsonl`

Write every predicted region, including unmatched predictions, for manual inspection.

### `predicted_groups.jsonl`

Write every predicted group and its member regions.

### `failures.jsonl`

Only write cases/pages/regions with violations or hard failures.

### `overlays/`

Write debug overlays when possible:

```text
overlays/page_001.overlay.png
overlays/page_001.matches.png
overlays/page_001.order.png
```

Overlay rules:

```text
Do not require overlays for scoring.
Do not let overlay failures break the benchmark unless --strict-overlays is passed.
Do not write overlays outside the run folder.
```

---

## 14. Script responsibilities

### `scripts/validate_fixtures.py`

Validate all benchmark files before evaluation.

Checks:

```text
benchmark folder exists
pages/ exists
labels/ exists
all label image paths exist
all JSON parses cleanly as UTF-8
schema_version is present
page_id values match filenames
box coordinates are four numeric values
boxes are inside or near image bounds
every extractable region has group_id and page_order
every group references existing region_id values
no duplicate region_id within a page
no duplicate group_id within a page
allowed orientation/kind values are used
ignore files reference the correct page_id
```

Exit nonzero on validation failure.

### `scripts/generate_synthetic_cases.py`

Generate deterministic synthetic benchmark pages and labels.

Requirements:

```text
no network calls
no model calls
stable output order
at least 12 synthetic pages
at least 60 labeled regions total
writes pages/, labels/, ignore_regions/, configs/, and README.md
uses a fixed random seed if randomness is used
```

Synthetic pages should cover:

```text
vertical speech bubbles
horizontal narration boxes
right-to-left manga reading order
multi-column vertical dialogue
small/furigana-like text
sound effects / SFX
sign text
page numbers and metadata that should not be extracted
credit/watermark-like text that should not be extracted
low-contrast text
text over light texture
nearby bubbles that must not be merged
one intended group split across multiple small regions
two separate groups close together
mixed Japanese/Latin noise
punctuation-only or ellipsis-like source text
```

Synthetic cases are not a replacement for real hand-labeled manga pages. They are a cheap harness sanity check and regression detector.

If Japanese-capable fonts are unavailable locally, the generator should still create valid images and labels, but it should record a warning in `benchmarks/synthetic/README.md`. Do not download fonts.

### `scripts/build_case_from_debug.py`

Optional utility for scaffolding real benchmark labels from prior project debug output.

Requirements:

```text
read-only access to a source debug/quality-run folder
no model calls
no network calls
copy only compact fixture data into source_extraction_autoresearch/benchmarks/cases/<name>/
never depend on the original debug folder at evaluation time
mark generated labels as draft/unverified
require human review before draft labels are used as benchmark ground truth
```

This script may use existing `.ocr.json` or `.debug.png` outputs to prefill boxes and text, but those generated labels are not authoritative until manually reviewed.

### `scripts/io_adapters.py`

Provide the project interface:

```python
def extract_source_page(image_path: str, page_config: dict | None = None) -> dict:
    """Return predicted text regions, OCR text, groups, order, and timing."""
```

Rules:

```text
Keep imports local and defensive.
Prefer direct imports from manga_local_translator.
Prefer `manga_local_translator.pipeline.prepare_page_for_translation(...)` and map its returned `PreparedPage` into the harness schema.
Do not use `process_image(...)`; it is outside the source-extraction-only scope.
Do not use the CLI fallback for scored project-mode benchmark runs. The CLI is a full pipeline entry point and can accidentally run translation, erase, render, or vision stages.
If direct imports are not viable, project mode should fail clearly. Existing `.ocr.json` files may be read only by `build_case_from_debug.py` to draft labels, not by the scored evaluator.
Do not call translation, routing, rendering, erase, Qwen, CAT, OPUS, MADLAD, Argos, or vision code.
Return a clear adapter error on missing OCR/detector dependencies.
```

### `scripts/normalize_text.py`

Pure text normalization and CER utilities only.

Requirements:

```text
no project imports
no model calls
no file writes
UTF-8 safe
unit-testable helper functions
```

### `scripts/match_regions.py`

Pure region matching and group comparison only.

Requirements:

```text
no project imports
no model calls
no file writes except through caller
one-to-one matching
stable deterministic tie-breaking
```

Tie-breaking order for candidate matches:

```text
1. higher match_score
2. higher IoU
3. higher text similarity
4. lower ground-truth region_id lexicographically
5. lower predicted region_id lexicographically
```

### `scripts/score.py`

Pure scoring functions only.

Requirements:

```text
no project imports
no model calls
no file writes except through caller
calculate all per-page and aggregate metrics
apply hard failure flags
return source_extraction_score and component metrics
```

### `scripts/eval_source_extraction.py`

Main benchmark runner.

Responsibilities:

```text
parse CLI args
validate benchmark unless --skip-validation is explicitly passed
load page images, labels, ignore regions, and configs
call io_adapters.extract_source_page for each page
match predictions to labels
score each page and aggregate summary
write run outputs
append one TSV row
update best.json only when the run passes hard guards and improves the best score
```

Required CLI args:

```text
--benchmark PATH
--output PATH
--results PATH
--run-id RUN_ID
```

Recommended CLI args:

```text
--experiment-name TEXT
--tests-ok
--overwrite-output
--adapter-smoke
--strict-overlays
--benchmark-set-name TEXT
--baseline-score FLOAT
```

Safety behavior:

```text
Refuse to write output outside source_extraction_autoresearch/runs/ unless --allow-external-output is passed.
Refuse to write results outside source_extraction_autoresearch/results/ unless --allow-external-results is passed.
Refuse to run if benchmark path is outside source_extraction_autoresearch/ unless --allow-external-benchmark is passed.
```

### `scripts/visualize_overlays.py`

Create optional debug overlays.

Recommended overlay conventions:

```text
ground-truth boxes
predicted boxes
matched pairs
missed regions
false positives
reading order labels
orientation labels
```

Do not make overlay generation part of the score.

### `scripts/compare_runs.py`

Compare two run folders and report changed metrics.

Required behavior:

```text
read summary.json from both runs
print score delta
print metric deltas
list pages with largest regressions
list pages with largest improvements
```

### `scripts/summarize_results.py`

Read `results.tsv` and print:

```text
best run
latest run
score trend
kept runs
failed hard-guard runs
top regressions by metric
```

---

## 15. Real benchmark cases

After the synthetic harness works, create a small hand-labeled benchmark set under:

```text
source_extraction_autoresearch/benchmarks/cases/core_v1/
```

Recommended composition:

```text
8 to 20 pages total
at least 100 extractable text regions
at least 20 non-extractable labeled regions
vertical and horizontal pages
clean bubbles and difficult bubbles
SFX/sign/narration cases
low-resolution or low-contrast pages
pages with close bubbles that should not merge
pages with multi-part groups that should merge
pages with page numbers, credits, and metadata to ignore
```

Use only pages that the repository owner has the right to store privately or commit. If benchmark pages are copyrighted and should not be committed, keep them in a private local checkout and document that in the benchmark README.

Do not let the evaluator depend on external folders once a benchmark case is created. Copy the needed page images and labels into the benchmark folder.

---

## 16. First experiments this loop should support

The harness should be designed to support these future experiments without changing scoring:

```text
adaptive OCR crop padding
crop upscaling / binarization / contrast preprocessing
CTD + visual candidate fusion
Tesseract fallback eligibility
manga-ocr acceptance/rejection heuristics
small-region / furigana handling
vertical text merge/split policy
duplicate suppression
metadata/page-number filtering
orientation classification
right-to-left reading order
multi-region group construction
stable line/block ID generation
```

These experiments should modify only the allowed project files from `PROGRAM.md`.

---

## 17. Baseline setup checklist

During setup, Codex should complete this checklist:

```text
[ ] Create source_extraction_autoresearch/ and required subfolders.
[ ] Copy this setup spec into source_extraction_autoresearch/CODEX_SETUP.md.
[ ] Write README.md with quick commands and scope warnings.
[ ] Write PROGRAM.md with objective, allowed edits, forbidden edits, commands, and keep/revert rules.
[ ] Write BENCHMARK_SCHEMA.md with label, ignore-region, config, and predicted-output schemas.
[ ] Write SCORING.md with the exact formula and metric definitions.
[ ] Write .gitignore.
[ ] Write templates/result_header.tsv with the exact header.
[ ] Create results/results.tsv with the exact header.
[ ] Create results/best.json.
[ ] Create script stubs with CLI help and clear TODO-safe behavior.
[ ] Implement validate_fixtures.py.
[ ] Implement normalize_text.py.
[ ] Implement match_regions.py.
[ ] Implement score.py.
[ ] Implement generate_synthetic_cases.py.
[ ] Implement eval_source_extraction.py enough to run adapter-smoke mode.
[ ] Implement io_adapters.py with project mode or clear dependency failure.
[ ] Run synthetic generation.
[ ] Run fixture validation.
[ ] Run adapter-smoke evaluation.
[ ] Append the first baseline row to results/results.tsv.
```

The setup is not complete until the following command succeeds in adapter-smoke mode:

```bash
python source_extraction_autoresearch/scripts/eval_source_extraction.py \
  --benchmark source_extraction_autoresearch/benchmarks/synthetic \
  --output source_extraction_autoresearch/runs/smoke_baseline \
  --results source_extraction_autoresearch/results/results.tsv \
  --run-id smoke_baseline \
  --adapter-smoke \
  --overwrite-output
```

The real project-mode baseline can be run after detector/OCR dependencies are available.

---

## 18. Acceptance criteria for setup PR

The setup PR is acceptable only if:

```text
all new files are under source_extraction_autoresearch/
no main project source files are edited
no .testing files are edited
no render_autoresearch or translation_routing_autoresearch files are edited
README.md explains the loop is isolated
PROGRAM.md contains clear keep/revert rules
BENCHMARK_SCHEMA.md and SCORING.md exist
results/results.tsv exists with the correct header
best.json exists
synthetic fixtures can be generated
fixtures can be validated
adapter-smoke benchmark can run and append one row
run outputs are written under source_extraction_autoresearch/runs/<run_id>/
```

Do not claim the source extraction code has been optimized during setup. The setup task only creates the isolated loop and a baseline harness.

---

## 19. Summary instruction for future autoresearch agents

Future optimization agents should read:

```text
source_extraction_autoresearch/PROGRAM.md
source_extraction_autoresearch/SCORING.md
source_extraction_autoresearch/BENCHMARK_SCHEMA.md
```

Then they should make one small project-code change at a time, run tests, run the benchmark, append exactly one result row, and keep the change only when `source_extraction_score` improves without violating hard guards.

The loop is separate from the main project. Keep it that way.
