# Codex setup spec: isolated render autoresearch loop for AITRanslator

You are Codex working inside the `AITRanslator` repository. Your task is to set up an isolated autoresearch loop for optimizing only the manga text rendering/layout path.

The loop must be separate from the main project. Do **not** put loop scripts, benchmark cases, run outputs, result logs, or generated reports into `.testing/`, `manga_local_translator/`, or any other existing project folder. Everything for the loop must live under one new root folder:

```text
render_autoresearch/
```

The main project is only an import target and, later, the code under test. The loop itself is a separate research harness.

---

## 1. Non-negotiable separation rule

Create this folder in the repository root:

```text
render_autoresearch/
```

All new loop assets must go inside it.

Allowed new files during setup:

```text
render_autoresearch/**
```

Forbidden setup locations:

```text
.testing/**
manga_local_translator/**
quality-runs/**
.model*/**
root-level helper scripts such as render_eval.py
root-level result files such as render_results.tsv
```

The setup phase must not edit main project source code. It may only read project files and create the isolated loop folder.

After setup, future optimization experiments may edit a very small project-code surface, but the benchmark harness, benchmark fixtures, result logs, and run artifacts must remain under `render_autoresearch/`.

---

## 2. Purpose of the loop

The loop should optimize rendering quality and rendering speed only.

Do not benchmark OCR, translation, detection, model loading, Qwen repair, or GUI behavior.

The benchmark must use frozen inputs:

```text
image.png
blocks.json
translations.json
case_config.json
```

A benchmark run should load those frozen artifacts, call the existing render/layout code, render the page, measure quality/speed, compute a score, and append one result row.

Lower score is better.

---

## 3. Project files Codex should inspect before implementation

Read these files to understand current function signatures and data classes:

```text
README.md
manga_local_translator/render.py
manga_local_translator/pipeline.py
manga_local_translator/config.py
manga_local_translator/erase.py
manga_local_translator/debug_report.py
manga_local_translator/detect_types.py
manga_local_translator/line_identity.py
.testing/tests/test_render.py
```

Do not copy large chunks of main-project logic into the harness. Import and call the project functions directly.

The render harness should be written so it still works if minor signatures change. Prefer a small adapter layer that inspects/imports the real project classes rather than hard-coding too much.

---

## 4. Required folder tree

Create this structure:

```text
render_autoresearch/
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

  results/
    results.tsv
    best.json
    .gitkeep

  runs/
    .gitkeep

  scripts/
    __init__.py
    render_eval.py
    score.py
    io_adapters.py
    generate_synthetic_cases.py
    make_case_from_debug.py
    compare_runs.py
    summarize_results.py

  templates/
    case_config.example.json
    blocks.example.json
    translations.example.json
    result_header.tsv
```

Put this setup spec itself inside the loop folder as:

```text
render_autoresearch/CODEX_SETUP.md
```

No setup files should be placed outside `render_autoresearch/`.

---

## 5. File responsibilities

### `render_autoresearch/README.md`

Explain the loop in human terms:

- It is isolated from the main project.
- It benchmarks only rendering/layout.
- It uses frozen images, frozen text blocks, and frozen translations.
- It writes append-only results.
- It does not run OCR, translation, detection, Qwen, or GUI code.
- It imports the main project as a library.

Include quick commands:

```bash
python render_autoresearch/scripts/generate_synthetic_cases.py \
  --output render_autoresearch/benchmarks/synthetic

python render_autoresearch/scripts/render_eval.py \
  --benchmark render_autoresearch/benchmarks/synthetic \
  --output render_autoresearch/runs/baseline \
  --results render_autoresearch/results/results.tsv \
  --run-id baseline

python render_autoresearch/scripts/summarize_results.py \
  --results render_autoresearch/results/results.tsv
```

### `render_autoresearch/PROGRAM.md`

This is the agent-facing autoresearch instruction file. It should tell a future coding agent how to run experiments.

It must include:

- Objective: minimize `render_score`.
- Scope: rendering/layout only.
- Primary editable project file in phase 1: `manga_local_translator/render.py`.
- Secondary editable project file in phase 2: `manga_local_translator/pipeline.py`, only for reusing render layouts/fits or reducing duplicate render work.
- Forbidden edits: benchmark fixtures, scoring code, result-log schema, OCR, detection, translation, debug warning definitions.
- Required tests before accepting a change.
- Required benchmark command.
- Keep/revert rules.

Use this decision rule:

```text
Keep a change only if:
1. render unit tests pass,
2. render benchmark completes,
3. render_score improves versus current best,
4. clipped_rate does not increase,
5. outside_page_rate does not increase,
6. outside_bubble_rate does not increase by more than 2%,
7. median_font_size does not drop by more than 1 px unless clipped_rate improves.
```

### `render_autoresearch/BENCHMARK_SCHEMA.md`

Define the fixture schema exactly.

Each benchmark case lives in its own folder:

```text
render_autoresearch/benchmarks/cases/<case_id>/
  image.png
  blocks.json
  translations.json
  case_config.json
  notes.md                # optional
```

Synthetic cases may live here:

```text
render_autoresearch/benchmarks/synthetic/<case_id>/
```

`blocks.json` schema:

```json
{
  "schema_version": 1,
  "case_id": "page_001_vertical_bubbles",
  "blocks": [
    {
      "id": "b001",
      "text": "original source text or stable placeholder",
      "box": [10, 20, 80, 120],
      "confidence": 99.0,
      "metadata": {
        "kind": "speech_bubble",
        "orientation": "vertical"
      }
    }
  ]
}
```

`translations.json` schema:

```json
{
  "schema_version": 1,
  "case_id": "page_001_vertical_bubbles",
  "translations": {
    "b001": "This is the English translation to render."
  }
}
```

`case_config.json` schema:

```json
{
  "schema_version": 1,
  "erase_mode": "white",
  "padding": 8,
  "render_expand": 2.2,
  "base_font_size": 28,
  "font_path": null,
  "notes": "Default render benchmark config."
}
```

The adapter may add extra translation-map keys internally if the current project lookup function expects source text, line identity, or block identity. Do not require fixture authors to know those internal lookup details.

### `render_autoresearch/SCORING.md`

Document the metric.

Primary metric:

```text
render_score
```

Lower is better.

Use this formula:

```text
render_score =
  10000 * clipped_rate
+  5000  * text_outside_render_box_rate
+  3000  * outside_page_rate
+  2500  * outside_bubble_rate
+  2000  * collision_rate
+  1500  * tiny_font_rate
+  1000  * excessive_line_count_rate
+   900  * cramped_fit_rate
+   800  * orphan_line_rate
+   700  * bad_wrap_rate
+   600  * split_word_rate
+   250  * mean_overflow_px
+    10  * mean_wrap_score
+     1  * p95_render_ms_per_block
```

Definitions:

```text
blocks_total                    number of render blocks across all pages
clipped_rate                    clipped_blocks / blocks_total
text_outside_render_box_rate    warning text_outside_render_box / blocks_total
outside_page_rate               warning render_box_outside_page / blocks_total
outside_bubble_rate             warning render_box_outside_bubble / blocks_total
collision_rate                  render-box collision count / max(1, blocks_total)
tiny_font_rate                  blocks with font_size <= 8 or warning tiny_font / blocks_total
excessive_line_count_rate       warning excessive_line_count / blocks_total
cramped_fit_rate                warning cramped_fit / blocks_total
orphan_line_rate                warning orphan_line / blocks_total
bad_wrap_rate                   warning bad_wrap / blocks_total
split_word_rate                 warning split_word or split_word_count > 0 / blocks_total
mean_overflow_px                mean(max(0, overflow_width) + max(0, overflow_height))
mean_wrap_score                 mean TextFit.wrap_score
p95_render_ms_per_block         95th percentile measured render/layout time per block
```

Hard failures:

```text
unit tests fail            reject run
render_eval.py crashes     reject run
any required output missing reject run
NaN or invalid metric      reject run
benchmark/scorer edited    reject experiment
```

Tie-breakers:

```text
1. lower clipped_rate
2. lower tiny_font_rate
3. higher median_font_size
4. lower p95_render_ms_per_block
5. smaller project-source diff
```

### `render_autoresearch/.gitignore`

Keep generated outputs isolated and avoid committing heavy images accidentally.

Recommended content:

```gitignore
runs/*
!runs/.gitkeep
results/*.tmp
results/*.lock
benchmarks/private/*
__pycache__/
*.pyc
.DS_Store
```

Do not ignore `results/results.tsv`; it is the append-only research log.

---

## 6. Required scripts

### `scripts/io_adapters.py`

Responsibilities:

1. Load `image.png` with OpenCV.
2. Load `blocks.json` and convert each block into the project's current `TextBlock` class.
3. Load `translations.json` and build a translation dictionary accepted by `lookup_translation`.
4. Load `case_config.json` and produce the values needed by render functions.
5. Provide small geometry helpers used by scoring, such as box area, intersection area, and collision counting.

Implementation notes:

- Inspect the real `TextBlock` constructor in `manga_local_translator.detect_types`.
- Support missing optional fields with safe defaults.
- Preserve each fixture block `id` in metadata if the project type supports it.
- Ensure the translation map works with `lookup_translation(translations, block, block.text)`. At minimum, map both `block_id -> translation` and `source_text -> translation` where possible.

### `scripts/score.py`

Responsibilities:

1. Convert per-block metrics into aggregate metrics.
2. Compute `render_score` exactly as documented in `SCORING.md`.
3. Validate that all metrics are finite numbers.
4. Return a dict suitable for JSON and TSV logging.

Expose functions like:

```python
def compute_case_metrics(block_metrics: list[dict]) -> dict: ...
def compute_run_metrics(case_metrics: list[dict]) -> dict: ...
def compute_render_score(metrics: dict) -> float: ...
```

Do not import or mutate project rendering code from `score.py`. It should operate on metrics only.

### `scripts/render_eval.py`

This is the main benchmark runner.

CLI:

```bash
python render_autoresearch/scripts/render_eval.py \
  --benchmark render_autoresearch/benchmarks/cases \
  --output render_autoresearch/runs/<run_id> \
  --results render_autoresearch/results/results.tsv \
  --run-id <run_id>
```

Optional flags:

```text
--limit N
--case CASE_ID
--overwrite-output
--no-append-results
--font-path PATH
--base-font-size INT
--render-expand FLOAT
--padding INT
```

Benchmark steps per case:

1. Load the frozen image.
2. Load frozen blocks.
3. Load frozen translations.
4. Load case config.
5. Build project `TextBlock` objects.
6. Import project functions:

```python
from manga_local_translator.render import plan_render_layouts, plan_text_fits, render_translations
from manga_local_translator.erase import erase_text
from manga_local_translator.debug_report import render_layout_warnings
```

7. Call `plan_render_layouts`.
8. Call `plan_text_fits`.
9. Call `render_layout_warnings` for each layout/fit pair.
10. Call `erase_text`.
11. Call `render_translations`.
12. Write rendered image.
13. Write debug overlay image if easy to implement.
14. Write per-case metrics JSON.
15. Aggregate run metrics.
16. Compute score.
17. Append one row to `results/results.tsv` unless `--no-append-results` is set.

Required output layout:

```text
render_autoresearch/runs/<run_id>/
  run_summary.json
  metrics.tsv
  cases/
    <case_id>/
      rendered.png
      debug_overlay.png          # optional but preferred
      case_metrics.json
      block_metrics.json
```

Each `block_metrics.json` entry should include:

```json
{
  "case_id": "page_001_vertical_bubbles",
  "block_id": "b001",
  "source_text": "...",
  "translated_text": "...",
  "source_box": [10, 20, 80, 120],
  "render_box": [5, 15, 100, 135],
  "bubble_box": null,
  "font_size": 14,
  "line_count": 3,
  "fit_status": "fit",
  "clipped": false,
  "widest_line": 74,
  "total_height": 48,
  "usable_width": 86,
  "usable_height": 106,
  "attempted_font_size": 18,
  "line_height": 16,
  "overflow_width": 0,
  "overflow_height": 0,
  "wrap_score": 0.0,
  "wrap_warnings": [],
  "split_word_count": 0,
  "orphan_line_count": 0,
  "warnings": [],
  "layout_ms": 0.0,
  "fit_ms": 0.0,
  "render_ms": 0.0
}
```

The timing values do not need to be perfect, but they must be measured consistently. Use `time.perf_counter()`.

### `scripts/generate_synthetic_cases.py`

Create a small deterministic synthetic benchmark so the harness can run before private manga fixtures exist.

CLI:

```bash
python render_autoresearch/scripts/generate_synthetic_cases.py \
  --output render_autoresearch/benchmarks/synthetic
```

Generate at least these cases:

```text
synthetic_001_short_bubbles
synthetic_002_tall_vertical_sources
synthetic_003_long_translations
synthetic_004_tiny_bubbles
synthetic_005_dense_collisions
synthetic_006_flat_horizontal_boxes
```

Each synthetic case should contain:

```text
image.png
blocks.json
translations.json
case_config.json
notes.md
```

Use simple white backgrounds, bubble-like outlines, and deterministic box coordinates. Do not require OCR/detection/translation models.

### `scripts/make_case_from_debug.py`

Create frozen cases from existing debug artifacts or a manually supplied image plus JSON.

CLI shape:

```bash
python render_autoresearch/scripts/make_case_from_debug.py \
  --image path/to/page.png \
  --debug-json path/to/debug_report.json \
  --output render_autoresearch/benchmarks/cases/page_001
```

The script should be conservative:

- Never overwrite existing cases unless `--overwrite` is passed.
- Extract render blocks and translations if present.
- If the debug JSON schema is unclear, write a partial case and print the missing fields.
- Do not place the new case outside `render_autoresearch/benchmarks/`.

### `scripts/compare_runs.py`

Compare two run summaries.

CLI:

```bash
python render_autoresearch/scripts/compare_runs.py \
  --before render_autoresearch/runs/baseline/run_summary.json \
  --after render_autoresearch/runs/experiment_001/run_summary.json
```

Print:

```text
render_score delta
clipped_rate delta
tiny_font_rate delta
outside_page_rate delta
outside_bubble_rate delta
median_font_size delta
p95_render_ms_per_block delta
keep/reject recommendation
```

### `scripts/summarize_results.py`

Read `results/results.tsv`, sort by `render_score`, and print the top runs.

CLI:

```bash
python render_autoresearch/scripts/summarize_results.py \
  --results render_autoresearch/results/results.tsv \
  --top 10
```

---

## 7. Result log schema

Create `render_autoresearch/results/results.tsv` with this exact header:

```tsv
run_id	timestamp_utc	git_commit	git_branch	git_dirty	benchmark_path	output_path	cases	blocks	unit_tests_ok	render_score	baseline_score	delta_score	clipped_rate	text_outside_render_box_rate	outside_page_rate	outside_bubble_rate	collision_rate	tiny_font_rate	excessive_line_count_rate	cramped_fit_rate	orphan_line_rate	bad_wrap_rate	split_word_rate	mean_overflow_px	mean_wrap_score	mean_font_size	median_font_size	p95_render_ms_per_block	mean_render_ms_per_block	kept	notes
```

Rules:

- Append one row per benchmark run.
- Never overwrite historical rows.
- Use ISO-8601 UTC timestamps.
- `git_dirty` must indicate whether uncommitted changes existed during the run.
- `kept` should default to `unknown` during raw benchmark execution. A later compare step may recommend `keep` or `reject`, but do not falsify history.
- `notes` should be short and shell-safe; avoid tabs inside notes.

Create `render_autoresearch/results/best.json` as:

```json
{
  "schema_version": 1,
  "best_run_id": null,
  "best_render_score": null,
  "updated_at_utc": null,
  "notes": "Updated by summarize_results.py or compare_runs.py. results.tsv remains the source of truth."
}
```

---

## 8. Benchmark command and acceptance commands

The setup should document these as the standard commands.

Run render unit tests:

```bash
python -m unittest discover -s .testing/tests -p test_render.py
```

On Windows PowerShell with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .testing\tests -p test_render.py
```

Generate synthetic benchmark cases:

```bash
python render_autoresearch/scripts/generate_synthetic_cases.py \
  --output render_autoresearch/benchmarks/synthetic
```

Run a benchmark:

```bash
python render_autoresearch/scripts/render_eval.py \
  --benchmark render_autoresearch/benchmarks/synthetic \
  --output render_autoresearch/runs/baseline \
  --results render_autoresearch/results/results.tsv \
  --run-id baseline \
  --overwrite-output
```

Summarize results:

```bash
python render_autoresearch/scripts/summarize_results.py \
  --results render_autoresearch/results/results.tsv \
  --top 10
```

---

## 9. Experiment loop for future agents

The setup should include this loop in `PROGRAM.md`.

```text
1. Read render_autoresearch/PROGRAM.md.
2. Check git status.
3. Run render unit tests.
4. Run the current benchmark and record baseline if needed.
5. Make one small hypothesis-driven change.
6. Edit only allowed project file(s).
7. Run render unit tests again.
8. Run render benchmark again.
9. Compare against current best.
10. Keep only if the decision rule passes.
11. Append notes to results.tsv through the benchmark/compare scripts.
12. Revert rejected project-code changes.
```

Initial experiment ideas to include:

```text
1. Cache fonts and repeated text-width measurements.
2. Reuse planned TextFit data during final drawing where possible.
3. Make font-size search faster without increasing clipping.
4. Improve candidate render-box scoring.
5. Improve punctuation/orphan-aware wrapping.
6. Improve narrow vertical-source layout behavior.
7. Improve collision handling for dense bubbles.
```

---

## 10. Guardrails against metric hacking

Codex must add these guardrails to `PROGRAM.md`:

```text
Do not improve the score by weakening warnings.
Do not edit render_autoresearch/scripts/score.py during optimization experiments.
Do not edit benchmark cases during optimization experiments.
Do not delete hard cases.
Do not change result-log history.
Do not skip blocks to reduce failure rates.
Do not run OCR, detection, translation, or model repair inside render_eval.py.
Do not compare runs on different benchmark folders unless the result row clearly states the benchmark path.
```

The benchmark should count missing or skipped rendered blocks as failures.

---

## 11. Implementation details for `render_eval.py`

Use this high-level pseudocode:

```python
def evaluate_case(case_dir, output_dir, overrides):
    image_bgr = load_image(case_dir / "image.png")
    block_specs = load_blocks(case_dir / "blocks.json")
    translations = load_translations(case_dir / "translations.json")
    case_config = load_case_config(case_dir / "case_config.json", overrides)

    blocks = build_project_text_blocks(block_specs)
    translation_map = build_project_translation_map(block_specs, translations, blocks)

    t0 = perf_counter()
    layouts = plan_render_layouts(
        image_bgr,
        blocks,
        render_expand=case_config.render_expand,
    )
    layout_ms_total = elapsed_ms(t0)

    t1 = perf_counter()
    fits = plan_text_fits(
        image_bgr,
        blocks,
        translation_map,
        font_path=case_config.font_path,
        base_font_size=case_config.base_font_size,
        render_expand=case_config.render_expand,
        render_layouts=layouts,
    )
    fit_ms_total = elapsed_ms(t1)

    erase_blocks = blocks
    cleaned = erase_text(
        image_bgr,
        erase_blocks,
        mode=case_config.erase_mode,
        padding=case_config.padding,
    )

    t2 = perf_counter()
    rendered = render_translations(
        cleaned,
        blocks,
        translation_map,
        font_path=case_config.font_path,
        base_font_size=case_config.base_font_size,
        render_expand=case_config.render_expand,
        render_layouts=layouts,
    )
    render_ms_total = elapsed_ms(t2)

    warnings = [
        render_layout_warnings(layout, fit, image_width=image_width, image_height=image_height)
        for layout, fit in zip(layouts, fits)
    ]

    block_metrics = build_block_metrics(...)
    case_metrics = compute_case_metrics(block_metrics)
    write_outputs(...)
    return case_metrics
```

Important:

- Use the existing project warning function, not a duplicate warning implementation.
- Use the existing project fitting/layout/rendering functions.
- Do not alter the image before measuring layout/fits except in the same way the project render path does.
- Use the same benchmark path for baseline and experiments.

---

## 12. Private fixture policy

If real manga pages are used, keep them under:

```text
render_autoresearch/benchmarks/private/
```

Do not commit private/copyrighted images unless the repository owner explicitly wants them committed. The `.gitignore` should ignore `benchmarks/private/*`.

The synthetic benchmark should be committed because it is deterministic and model-free.

---

## 13. Setup completion checklist

The setup is complete only when all of these are true:

```text
[ ] render_autoresearch/ exists at repository root.
[ ] No loop files were created outside render_autoresearch/.
[ ] README.md explains the loop and quick commands.
[ ] CODEX_SETUP.md contains this setup spec.
[ ] PROGRAM.md contains future-agent experiment rules.
[ ] BENCHMARK_SCHEMA.md defines case files.
[ ] SCORING.md defines render_score and metrics.
[ ] results/results.tsv exists with the exact header.
[ ] results/best.json exists.
[ ] scripts/render_eval.py exists and can run on synthetic cases.
[ ] scripts/score.py computes render_score.
[ ] scripts/io_adapters.py converts fixture JSON to project objects.
[ ] scripts/generate_synthetic_cases.py creates at least six cases.
[ ] scripts/summarize_results.py prints ranked results.
[ ] render unit tests still pass or failures are reported honestly.
[ ] A baseline synthetic benchmark run has been attempted.
[ ] git status shows setup changes are confined to render_autoresearch/.
```

If a command fails because the current project API differs from this spec, fix the adapter or document the exact mismatch in `render_autoresearch/README.md`. Do not move the loop into the main project to work around import issues.

---

## 14. Expected final response from Codex after setup

When done, report:

```text
Created isolated loop folder: render_autoresearch/
Created benchmark scaffold: yes/no
Created synthetic cases: yes/no
Render unit tests: pass/fail/not run
Baseline benchmark: pass/fail/not run
Results log: render_autoresearch/results/results.tsv
Run output: render_autoresearch/runs/<run_id>
Files changed outside render_autoresearch/: none/list them
Next recommended experiment: <one sentence>
```

Be honest about failures. A partially working isolated harness is better than silently mixing research files into the main project.
