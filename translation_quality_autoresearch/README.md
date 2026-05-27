# Translation Quality Autoresearch

This is an isolated replay and trace evaluation harness for the AITRanslator translation-agent stage:

```text
validated OCR source blocks + context -> translator agents -> final English translation
```

It does not evaluate detection, OCR, source extraction, grouping, erase, rendering, inpainting, GUI behavior, or model installation. All benchmark data, scripts, scores, traces, reports, prompts, and run outputs live under `translation_quality_autoresearch/`. Deleting this folder leaves the main project behavior unchanged.

## Replay Evaluation

PowerShell:

```powershell
# Validate benchmark schemas
.\.venv\Scripts\python.exe translation_quality_autoresearch\scripts\validate_benchmark.py `
  --benchmark translation_quality_autoresearch\benchmark

# Run unit tests for the harness
.\.venv\Scripts\python.exe -m unittest discover -s translation_quality_autoresearch\tests

# Run replay-mode evaluation with frozen outputs
.\.venv\Scripts\python.exe translation_quality_autoresearch\scripts\eval_translation_quality.py `
  --mode replay `
  --cases translation_quality_autoresearch\benchmark\cases.jsonl `
  --references translation_quality_autoresearch\benchmark\references.jsonl `
  --frozen-outputs translation_quality_autoresearch\benchmark\frozen_agent_outputs.jsonl `
  --glossary translation_quality_autoresearch\benchmark\glossary.json `
  --output translation_quality_autoresearch\runs\baseline_replay `
  --results translation_quality_autoresearch\results\translation_quality_results.tsv `
  --run-id baseline_replay `
  --overwrite-output
```

Unix-style:

```bash
python translation_quality_autoresearch/scripts/validate_benchmark.py \
  --benchmark translation_quality_autoresearch/benchmark

python -m unittest discover -s translation_quality_autoresearch/tests

python translation_quality_autoresearch/scripts/eval_translation_quality.py \
  --mode replay \
  --cases translation_quality_autoresearch/benchmark/cases.jsonl \
  --references translation_quality_autoresearch/benchmark/references.jsonl \
  --frozen-outputs translation_quality_autoresearch/benchmark/frozen_agent_outputs.jsonl \
  --glossary translation_quality_autoresearch/benchmark/glossary.json \
  --output translation_quality_autoresearch/runs/baseline_replay \
  --results translation_quality_autoresearch/results/translation_quality_results.tsv \
  --run-id baseline_replay \
  --overwrite-output
```

## Live Agent Generation

Live generation writes traces only. Evaluation of those traces is deterministic and does not re-call models.

```powershell
# Generate live agent candidates, if local models are available
.\.venv\Scripts\python.exe translation_quality_autoresearch\scripts\run_translation_agents.py `
  --cases translation_quality_autoresearch\benchmark\cases.jsonl `
  --agents opus,qwen_block,qwen_page,cat `
  --output translation_quality_autoresearch\runs\live_candidates\agent_traces.jsonl

# Evaluate a live trace
.\.venv\Scripts\python.exe translation_quality_autoresearch\scripts\eval_translation_quality.py `
  --mode live-trace `
  --cases translation_quality_autoresearch\benchmark\cases.jsonl `
  --references translation_quality_autoresearch\benchmark\references.jsonl `
  --traces translation_quality_autoresearch\runs\live_candidates\agent_traces.jsonl `
  --glossary translation_quality_autoresearch\benchmark\glossary.json `
  --output translation_quality_autoresearch\runs\live_eval `
  --results translation_quality_autoresearch\results\translation_quality_results.tsv `
  --run-id live_eval `
  --overwrite-output
```

## Inspecting Runs

Each run writes the handoff artifacts expected by optimization agents:

```text
summary.json
summary.md
case_results.jsonl
traces.jsonl
failures.jsonl
artifacts/run_manifest.json
```

`candidate_scores.jsonl`, `case_decisions.jsonl`, and `failure_table.tsv` are still written for backward-compatible inspection. The append-only result log is `results/translation_quality_results.tsv`; pass `--no-append-results` for local smoke runs that should not be recorded.

Add benchmark cases by editing or creating JSONL files under `benchmark/`, then run `scripts/validate_benchmark.py`. Do not tune prompts, rules, or rerankers against hidden label edits. Future optimization agents must not modify benchmark files, scorer code, tests, or result logs unless the explicit task is to improve the harness itself.

## Agent Readiness

For agent handoff with manually translated Codex references, use the versioned benchmarks:

```powershell
.\.venv\Scripts\python.exe translation_quality_autoresearch\scripts\validate_benchmark.py `
  --benchmark translation_quality_autoresearch\benchmark_v1

.\.venv\Scripts\python.exe translation_quality_autoresearch\scripts\report_benchmark_coverage.py `
  --cases translation_quality_autoresearch\benchmark_v1\cases.jsonl `
  --references translation_quality_autoresearch\benchmark_v1\references.jsonl `
  --human-gold translation_quality_autoresearch\benchmark_v1\human_gold.jsonl `
  --readiness-profile machine_assisted

.\.venv\Scripts\python.exe translation_quality_autoresearch\scripts\eval_translation_quality.py `
  --mode replay `
  --cases translation_quality_autoresearch\benchmark_v1\cases.jsonl `
  --references translation_quality_autoresearch\benchmark_v1\references.jsonl `
  --frozen-outputs translation_quality_autoresearch\benchmark_v1\frozen_agent_outputs.jsonl `
  --glossary translation_quality_autoresearch\benchmark_v1\glossary.json `
  --output translation_quality_autoresearch\runs\v1_baseline_replay `
  --results translation_quality_autoresearch\results\translation_quality_results_v1.tsv `
  --run-id v1_baseline_replay `
  --overwrite-output
```

Use `benchmark_holdout_v1` only as a final gate after a candidate looks good on `benchmark_v1`. Both v1 benchmarks use `reference_policy: manual_codex_translation`; they are suitable for early optimization handoff, not final product-quality claims without native-speaker/human review.

Before assigning autonomous optimization work, read `AGENT_START.md`, `BENCHMARK_READINESS.md`, and run a coverage report:

```powershell
.\.venv\Scripts\python.exe translation_quality_autoresearch\scripts\report_benchmark_coverage.py `
  --cases translation_quality_autoresearch\benchmark\cases.jsonl,translation_quality_autoresearch\benchmark\adversarial_cases.jsonl,translation_quality_autoresearch\benchmark\page_context_cases.jsonl `
  --output translation_quality_autoresearch\runs\benchmark_coverage.md
```

The checked-in benchmark is a seed/smoke benchmark. It is intentionally small and should not be treated as sufficient for fully autonomous prompt or rule optimization. See `benchmark/VERSIONING.md` for the benchmark expansion target and versioning rules.

Use `scripts/compare_runs.py` after an experiment to compare the candidate summary against the baseline summary. The comparison recommends `KEEP` only when score improves and guardrail metrics do not regress.
