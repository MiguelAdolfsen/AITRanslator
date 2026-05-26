# Source Extraction Autoresearch

This folder is an isolated research loop for Japanese source extraction quality in AITRanslator.
It benchmarks only:

- text detection
- OCR crop preparation and OCR text
- bad-fragment filtering
- orientation classification
- grouping into source units
- page and manga reading order

It does not run translation, routing, Qwen, CAT, OPUS, MADLAD, Argos, rendering, erase, inpainting, GUI code, or model setup. The harness imports the main project as a library and calls the source-extraction path through `scripts/io_adapters.py`.

Benchmarks use frozen page images plus UTF-8 hand-labeled source-region JSON. Results are append-only and are meant to help future agents decide whether a small extraction change should be kept or reverted.

Quick commands:

```bash
python source_extraction_autoresearch/scripts/generate_synthetic_cases.py \
  --output source_extraction_autoresearch/benchmarks/synthetic

python source_extraction_autoresearch/scripts/validate_fixtures.py \
  --benchmark source_extraction_autoresearch/benchmarks/synthetic

python source_extraction_autoresearch/scripts/eval_source_extraction.py \
  --benchmark source_extraction_autoresearch/benchmarks/synthetic \
  --output source_extraction_autoresearch/runs/baseline \
  --results source_extraction_autoresearch/results/results.tsv \
  --run-id baseline \
  --run-tests

python source_extraction_autoresearch/scripts/summarize_results.py \
  --results source_extraction_autoresearch/results/results.tsv
```

Useful loop maintenance commands:

```bash
python source_extraction_autoresearch/scripts/rebuild_best_index.py \
  --results source_extraction_autoresearch/results/results.tsv

python source_extraction_autoresearch/scripts/benchmark_coverage.py \
  --benchmark source_extraction_autoresearch/benchmarks/cases/local_real_diverse_v2_frozen

python source_extraction_autoresearch/scripts/report_failures.py \
  --run source_extraction_autoresearch/runs/local_real_diverse_v2_current

python source_extraction_autoresearch/scripts/run_benchmark_matrix.py \
  --include-historical
```

Each evaluator output folder contains `manifest.json`, `summary.json`, per-page metrics, region matches,
predicted regions/groups, overlays, and failure rows. New failure rows include structured false-positive
details, group/order diagnostics, and score breakdowns. Decision runs require harness-owned tests through
`--run-tests` or a matrix-supplied test context.

On Windows in this project, prefer the repo virtualenv when present:

```powershell
.\.venv\Scripts\python.exe source_extraction_autoresearch\scripts\generate_synthetic_cases.py `
  --output source_extraction_autoresearch\benchmarks\synthetic
```

Setup smoke command:

```bash
python source_extraction_autoresearch/scripts/eval_source_extraction.py \
  --benchmark source_extraction_autoresearch/benchmarks/synthetic \
  --output source_extraction_autoresearch/runs/smoke_baseline \
  --results source_extraction_autoresearch/results/results.tsv \
  --run-id smoke_baseline \
  --adapter-smoke \
  --no-decision-run \
  --overwrite-output
```
