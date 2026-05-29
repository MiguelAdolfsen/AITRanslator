# Autoresearch

## Start Here

Autoresearch harnesses are isolated benchmark loops. Always read the relevant harness `PROGRAM.md` before editing benchmark-related code or interpreting results.

## Harness Map

| Folder | Purpose | Durable docs |
|---|---|---|
| `render_autoresearch/` | Rendering/layout benchmark with frozen synthetic fixtures. | `README.md`, `PROGRAM.md`, `BENCHMARK_SCHEMA.md`, `SCORING.md` |
| `source_extraction_autoresearch/` | Source extraction/OCR-region benchmark and scoring. | `README.md`, `PROGRAM.md`, `CODEX_SETUP.md`, `BENCHMARK_SCHEMA.md`, `SCORING.md` |
| `translation_routing_autoresearch/` | Translation candidate routing/selection benchmark. | `README.md`, `PROGRAM.md`, `CODEX_SETUP.md`, `BENCHMARK_SCHEMA.md`, `SCORING.md` |
| `cat_response_autoresearch/` | CAT response quality and schema benchmark. | `README.md`, `PROGRAM.md`, `BENCHMARK_SCHEMA.md`, `SCORING.md` |
| `autoresearch_docs/` | Historical/setup briefs and research notes. | Markdown setup and summary docs |

## Generated Outputs

Treat these as generated evidence, not source of truth:

- harness `runs/` folders,
- result logs unless the harness says they are append-only durable records,
- `.testing/output/`,
- `.testing/work/`,
- `quality-runs/`,
- private benchmark folders,
- local model folders.

## Guardrails

- Do not change fixtures, scoring, warning definitions, or result history to improve scores unless the harness program explicitly allows it.
- Do not compare scores across different benchmark folders without making the benchmark identity explicit.
- Do not skip hard cases to improve metrics.
- Do not run unrelated pipeline stages inside isolated harnesses.
- Revert rejected project-code changes while preserving unrelated user changes.

## Render Harness

For render/layout work, read `render_autoresearch/PROGRAM.md`. It limits primary edits to `manga_local_translator/render.py` and, for specific reuse/performance cases, `pipeline.py`.

Typical checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .testing\tests -p test_render.py
```

```powershell
.\.venv\Scripts\python.exe render_autoresearch\scripts\render_eval.py `
  --benchmark render_autoresearch\benchmarks\synthetic `
  --output render_autoresearch\runs\<run_id> `
  --results render_autoresearch\results\results.tsv `
  --run-id <run_id> `
  --overwrite-output
```

## Choosing A Harness

| Goal | Harness |
|---|---|
| Better rendered text placement/wrapping | `render_autoresearch/` |
| Better detection/OCR/source region extraction | `source_extraction_autoresearch/` |
| Better translation selection/routing among candidates | `translation_routing_autoresearch/` |
| Better CAT response filtering and scoring | `cat_response_autoresearch/` |
| Whole-pipeline translation quality summaries | `manga_local_translator.quality_eval` |

