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
  --run-id baseline

python source_extraction_autoresearch/scripts/summarize_results.py \
  --results source_extraction_autoresearch/results/results.tsv
```

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
  --overwrite-output
```

