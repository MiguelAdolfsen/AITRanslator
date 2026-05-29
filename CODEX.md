# Agent Guide

## Start Here

This is a local-first Japanese manga image translator. It detects text regions in page images, OCRs Japanese text, filters and groups blocks, translates to English, erases the original text, renders translated text back onto the image, and writes debug/cache artifacts.

Read this file first, then use [docs/agent-wiki/README.md](docs/agent-wiki/README.md) as the deeper map.

## Environment And Tests

Use the repo virtual environment when running Python. On Windows/PowerShell, prefer the explicit interpreter path:

```powershell
.\.venv\Scripts\python.exe -m manga_local_translator --help
```

The default test entrypoint is the PowerShell runner under `.testing`:

```powershell
.\.testing\run_tests.ps1
```

For focused unittest runs, keep the leading dot in `.testing`:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .testing\tests -p test_render.py
```

Use `.\.testing\run_tests.ps1 -WithSmoke` only when local smoke inputs and required model/runtime dependencies are available.

## 60-Second Map

| Area | Primary files | Read first | Run/check |
|---|---|---|---|
| CLI behavior | `manga_local_translator/cli.py`, `manga_local_translator/config.py` | [configuration](docs/agent-wiki/configuration.md) | `.\.venv\Scripts\python.exe -m manga_local_translator --help` |
| GUI behavior | `manga_local_translator/gui.py` | [runtime flows](docs/agent-wiki/runtime-flows.md) | `.\.venv\Scripts\python.exe -m manga_local_translator.gui` |
| Pipeline flow | `manga_local_translator/pipeline.py` | [architecture](docs/agent-wiki/architecture.md) | targeted unit tests plus CLI smoke |
| Detection/OCR | `detect_ocr.py`, `ctd_detector.py`, `source_extraction.py` | [data contracts](docs/agent-wiki/data-contracts.md) | `test_detect_ocr_heuristics.py`, source extraction tests |
| Translation | `translate.py`, `hf_translators.py`, `qwen_translator.py`, `translation_rules.py` | [configuration](docs/agent-wiki/configuration.md) | translation, Qwen, CAT, Hy-MT2, and routing tests |
| Qwen/vision | `qwen_*`, `vision_*` | [runtime flows](docs/agent-wiki/runtime-flows.md) | Qwen and vision unit tests |
| Rendering | `render.py`, `erase.py`, `debug_report.py` | [debugging](docs/agent-wiki/debugging.md) | `.\.venv\Scripts\python.exe -m unittest discover -s .testing\tests -p test_render.py` |
| Quality eval | `quality_eval.py`, `review_report.py` | [testing](docs/agent-wiki/testing.md) | `.\.venv\Scripts\python.exe -m manga_local_translator.quality_eval --help` |
| Autoresearch | `*_autoresearch/`, `autoresearch_docs/` | [autoresearch](docs/agent-wiki/autoresearch.md) | harness-specific `PROGRAM.md` commands |

## Recommended Read Order

1. [Wiki index](docs/agent-wiki/README.md)
2. [Architecture](docs/agent-wiki/architecture.md)
3. [Runtime flows](docs/agent-wiki/runtime-flows.md)
4. [Configuration](docs/agent-wiki/configuration.md)
5. [Testing](docs/agent-wiki/testing.md)

For data shape changes, read [data contracts](docs/agent-wiki/data-contracts.md). For failures, read [debugging](docs/agent-wiki/debugging.md). For benchmark loops, read [autoresearch](docs/agent-wiki/autoresearch.md) and the relevant harness `PROGRAM.md`.

## Safety Notes

- Treat `.models/`, `.testing/output/`, `.testing/work/`, `quality-runs/`, `mangafolder/`, and large harness `runs/` folders as generated or local data, not source of truth.
- Do not weaken benchmark scoring, warnings, fixtures, or guardrails to make results look better.
- Do not assume model files are present. Runtime supports local paths and first-time downloads for some backends.
- Keep CLI flags, `PipelineConfig`, README examples, and this wiki in sync when behavior changes.
- Prefer narrow tests for focused changes, then use `.\.testing\run_tests.ps1` for broader confidence.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues for `MiguelAdolfsen/AITRanslator`. See `docs/agents/issue-tracker.md`.

### Triage labels

This repo uses the default triage label vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repo with one root `CONTEXT.md`; ADRs live under `docs/adr/` when present. See `docs/agents/domain.md`.
