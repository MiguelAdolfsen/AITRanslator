# Source Extraction Autoresearch Evaluation

Date: 2026-05-26

Upstream reference: [karpathy/autoresearch](https://github.com/karpathy/autoresearch), inspected at commit `228791fb499afffb54b46200aca536f79142f117`.

Status note: this document started as an evaluation and improvement plan. Most harness recommendations below have now been implemented. Current agent instructions live in `PROGRAM.md`, `KEEP_EXPERIMENTING.md`, `SCORING.md`, and `README.md`.

Implemented since this evaluation:

- Per-benchmark best sidecars: `results/best.<benchmark>.json`.
- Strict decision-run enforcement in `eval_source_extraction.py`.
- Harness-owned tests through `--run-tests` or matrix-supplied test context.
- `source_extraction_score` is quality-only; timing is a tie-breaker.
- Dedicated source extraction API: `manga_local_translator/source_extraction.py`.
- Failure reporting includes false positives, group failures, reading-order inversions, page score breakdowns, and timing-only pages.
- Benchmark coverage and matrix runner scripts exist.

## Upstream Pattern

`karpathy/autoresearch` is intentionally narrow:

- One fixed preparation/evaluation file: `prepare.py`.
- One editable experiment file: `train.py`.
- One human-authored research program: `program.md`.
- One primary metric: `val_bpb`, lower is better.
- One append-only experiment log: `results.tsv`.
- A tight keep/discard loop: run for a fixed wall-clock budget, record the metric, keep the commit only if it improves.

The important idea for this project is not the LLM training code. It is the operating contract: small editable surface, frozen evaluator, comparable runs, and a result log that makes keep/revert decisions mechanical.

## Current Local Shape

`source_extraction_autoresearch/` already follows a lot of that pattern:

- `PROGRAM.md` defines objective, scope, forbidden files, required tests, benchmark commands, and keep criteria.
- `SCORING.md` defines a single lower-is-better `source_extraction_score`.
- `eval_source_extraction.py` writes run artifacts, overlays, JSONL diagnostics, and a TSV row.
- `BENCHMARK_SCHEMA.md` gives a project-specific label/prediction contract.
- `benchmarks/synthetic/` gives deterministic smoke coverage.
- Local frozen real cases exist:
  - `local_real_diverse_v2_frozen`: 24 pages.
  - `local_spy_short_v1_frozen`: 8 pages.
  - `local_hard_pages_v1_frozen`: 12 pages.

This was a solid start. The main gaps identified below have been addressed in the current harness unless marked as ongoing.

## Key Gaps

1. **Best-run tracking is not benchmark-aware.** Status: implemented.

   `results/results.tsv` still contains historical rows, but current best tracking is per benchmark via `results/best.<benchmark>.json`. Smoke and draft rows are non-decision rows.

2. **The primary real benchmark is documented but not yet driving results.** Status: implemented.

   `local_real_diverse_v2_frozen/` now has a baseline row and a per-benchmark best sidecar.

3. **The evaluator logs guards but does not enforce most of them.** Status: implemented.

   Decision runs are strict. Use `--run-tests` for single decision runs or the matrix runner for shared test context. Old trust-based test flags are deprecated.

4. **The editable surface is wider than upstream's equivalent.** Status: partly inherent, API isolation implemented.

   This project still needs a wider editable surface than upstream, but the adapter now calls `manga_local_translator.source_extraction.extract_source_page(...)`.

5. **Timing currently has too much influence after quality improves.** Status: implemented.

   `source_extraction_score` is quality-only. Timing is reported as `timing_score_component` and used as a tie-breaker.

6. **Reproducibility metadata is incomplete.** Status: mostly implemented.

   New runs write `manifest.json` with command, Python/platform info, git state, benchmark fingerprint, row data, best path, adapter mode, and validation/test state.

7. **Diagnostics are useful but not yet decision-oriented.** Status: implemented.

   `report_failures.py` now reports quality-ranked pages, slow pages, false positives, OCR/crop issues, group failures, and reading-order inversions.

8. **Fixture validation is structural, not semantic.** Status: coverage reporting implemented; threshold enforcement remains optional.

   `benchmark_coverage.py` reports coverage by kind, orientation, difficulty tags, group sizes, ignore files, and extractable mix.

## Current Improvement Plan Status

### 1. Make results benchmark-aware

Implemented.

- Current best records are benchmark-specific, for example `results/best.local_real_diverse_v2_frozen.json`.
- Adapter-smoke and draft benchmarks are non-decision rows and do not update best sidecars.
- `summarize_results.py` reports benchmark groups and highlights matrix runs.
- New rows include benchmark fingerprint, adapter mode, and decision-run fields.

### 2. Enforce the program contract in code

Implemented.

- Decision-run preflight fails on dirty forbidden files unless an explicit override is passed.
- Single benchmark decision runs use `--run-tests`; matrix runs execute tests once and pass matrix test context into child evaluations.
- Decision runs fail when validation is skipped, adapter smoke is used, or the benchmark path/name contains `_draft`.
- Baseline and delta are computed from the current per-benchmark best sidecar.

### 3. Restore the intended benchmark gate

Use the matrix runner for real experiments:

1. Synthetic smoke: quick harness regression only.
2. `local_real_diverse_v2_frozen`: primary keep/revert benchmark.
3. `local_spy_short_v1_frozen`: historical regression check.
4. `local_hard_pages_v1_frozen`: historical hard-page regression check.

Keep a change only when it improves v2 and does not violate hard guardrails on the two historical sets. Synthetic catches obvious breakage but does not decide the winner.

Preferred command:

```powershell
.\.venv\Scripts\python.exe source_extraction_autoresearch\scripts\run_benchmark_matrix.py `
  --include-historical `
  --overwrite-output
```

For a single decision benchmark, use `eval_source_extraction.py --run-tests`.

### 4. Isolate source extraction behind a dedicated project API

Implemented. The project-level function is:

```text
manga_local_translator/source_extraction.py
  extract_source_page(image_path, config) -> SourceExtractionResult
```

The autoresearch adapter calls that API. The result exposes raw OCR regions, accepted source regions, grouped render/source units, page order, orientation, filters applied, and timing. It does not touch translation, rendering, erase, vision, cache, or output image paths.

This gives the local loop an upstream-like "single experiment surface" even if the implementation spans multiple files internally.

### 5. Split quality score from performance score

Implemented. `source_extraction_score` equals `source_extraction_quality_score`. Timing is reported as `p95_extraction_ms_per_page`, `mean_extraction_ms_per_page`, and `timing_score_component`, and is used as a later best-run tie-breaker.

### 6. Add failure-focused reporting

Implemented. `scripts/report_failures.py` reads a run folder and emits:

- Top pages by contribution to total score.
- Top slow pages by timing.
- Missed extractable regions, with page id, region id, kind, orientation, and difficulty tags.
- Destructive false positives, with overlap reason and nearest label.
- OCR severe errors, with reference text, OCR text, and CER.
- Grouping errors, with predicted/ground-truth member ids.
- Reading order inversions.

This is the biggest practical improvement for agent iteration speed.

### 7. Add benchmark coverage reporting

Implemented as `scripts/benchmark_coverage.py`. It reports:

- Pages, regions, extractable regions, non-extractable regions.
- Counts by `kind`.
- Counts by `orientation`.
- Counts by `difficulty_tags`.
- Extractable/non-extractable ratio.
- Group-size distribution.
- Pages with empty ignore/config files.

Use this to decide whether new labels add signal before freezing them. Coverage threshold enforcement remains optional future work.

### 8. Tighten run identity and reproducibility

Implemented. Each run writes a manifest and includes key fields in the TSV:

- `run_command`
- `benchmark_fingerprint`
- `adapter_mode`
- `project_config`
- `python_version`
- `dependency_snapshot`
- `ocr_engine`
- `detector`
- `model_paths_or_versions`
- `dirty_files_before_run`
- `dirty_files_after_run`

Decision rows should ideally correspond to a commit or a clearly named dirty experiment patch. Otherwise future agents cannot reconstruct what was actually tested.

### 9. Keep smoke, draft, and decision artifacts separate

Implemented in the main evaluator path.

- Adapter-smoke rows are marked non-decision.
- Draft benchmark experiments are marked non-decision and cannot be decision runs.
- Decision preflight fails if `_draft` appears in the benchmark path or name.

### 10. Prioritize extraction experiments from current evidence

Once v2 baseline rows exist, choose experiments from the actual failure mix. Based on the current logged synthetic and v1 rows, likely first targets are:

- False-positive filtering without increasing missed dialogue.
- Undermerge/overmerge behavior on hard pages.
- Reading-order stability after grouping changes.
- OCR crop changes only when they improve CER without adding misses or false positives.
- Timing improvements only after quality metrics are stable.

## Remaining Work

The infrastructure plan is implemented. Remaining work is extraction-quality iteration, not harness plumbing:

- Run the matrix before and after each extraction change.
- Use `report_failures.py` to pick one failure class at a time.
- Improve false-positive filtering without increasing missed dialogue.
- Improve undermerge/overmerge behavior on hard pages.
- Stabilize reading order after grouping changes.
- Change OCR crop logic only when it improves CER without adding misses or destructive false positives.
- Treat timing as a tie-breaker after quality metrics are stable.
