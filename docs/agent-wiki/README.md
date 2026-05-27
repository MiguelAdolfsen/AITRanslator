# Agent Wiki

## Start Here

This wiki is for agents that need a quick but accurate project overview before editing code. It is repo-native Markdown; there is no generated docs site.

The main app is `manga_local_translator`. It translates Japanese manga page images locally through this flow:

```text
input image/folder
  -> text detection
  -> OCR
  -> text filtering
  -> grouping and reading order
  -> translation
  -> optional Qwen critic/fallback/vision
  -> erase original text
  -> render English text
  -> debug reports, cache files, and translated images
```

## Wiki Pages

| Page | Use it for |
|---|---|
| [architecture.md](architecture.md) | Subsystem map and source anchors. |
| [runtime-flows.md](runtime-flows.md) | CLI, GUI, hybrid Qwen/CAT, render-only, and resume/cache flows. |
| [configuration.md](configuration.md) | `PipelineConfig`, CLI flags, model paths, env vars, and glossary behavior. |
| [data-contracts.md](data-contracts.md) | Core dataclasses, cache stages, debug reports, review outputs, and vision artifacts. |
| [testing.md](testing.md) | Test runner, unit test categories, quality eval, and render benchmark checks. |
| [debugging.md](debugging.md) | Logs, `.debug.png`, `.ocr.json`, common failure paths, and diagnosis commands. |
| [autoresearch.md](autoresearch.md) | Harness purposes, durable docs, generated outputs, and edit guardrails. |
| [maintenance.md](maintenance.md) | How to keep docs synchronized with code changes. |

Machine-readable page metadata lives in [wiki-manifest.json](wiki-manifest.json).

## Source Of Truth

The durable source is the Python package, root docs, test scripts, and harness program docs. Generated outputs and local caches are evidence, not policy.

| Durable | Generated/local |
|---|---|
| `manga_local_translator/` | `.models/` |
| `README.md` | `.testing/output/` |
| `.testing/README.md` and `.testing/tests/` | `.testing/work/` |
| `render_autoresearch/PROGRAM.md` and similar harness docs | `quality-runs/` |
| `translation_glossary.example.json` | harness `runs/` outputs |

## Local Environment Assumptions

- Commands in this wiki are Windows PowerShell-friendly because the primary local workflow uses PowerShell.
- Prefer `.\.venv\Scripts\python.exe` when a virtual environment exists; otherwise use `python` and let the active environment decide.
- Local smoke inputs such as `.testing/input/` and `mangafolder/` may be absent in a fresh clone.
- Model folders such as `.models/` are optional local assets. Do not assume detectors, translators, or Qwen/CAT/Vision models are already downloaded.
- Network access may be needed for first-time model downloads, but normal project behavior should remain local-first once assets are present.
- Generated work and result folders are useful for diagnosis, but do not treat them as authoritative behavior definitions.

## Common Tasks

| Task | Read first | Source anchors | Verification |
|---|---|---|---|
| Add/change CLI option | [configuration](configuration.md) | `cli.py`, `config.py`, `gui.py` if surfaced in UI | `python -m manga_local_translator --help` |
| Change page processing | [runtime flows](runtime-flows.md) | `pipeline.py`, `page_cache.py`, `page_types.py` | focused unit tests and CLI smoke |
| Improve OCR filtering | [data contracts](data-contracts.md) | `detect_ocr.py`, `text_filter.py`, `source_extraction.py` | OCR/source extraction tests |
| Improve translation routing | [architecture](architecture.md) | `translate.py`, `translation_rules.py`, `translation_evidence.py`, Qwen/CAT files | translation, Qwen, CAT tests |
| Improve render layout | [autoresearch](autoresearch.md) | `render.py`, `erase.py`, `debug_report.py` | render unit tests and render harness |
| Investigate bad output | [debugging](debugging.md) | generated `.ocr.json`, `.debug.png`, log | reproduce with `--debug` |
