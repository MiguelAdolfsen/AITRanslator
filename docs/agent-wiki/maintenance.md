# Wiki Maintenance

## Start Here

Update this wiki whenever agent-facing behavior changes. The goal is fast orientation, not exhaustive API documentation.

## Update Checklist

| Change | Docs to update |
|---|---|
| New CLI flag | `README.md`, `AGENTS.md` if task routing changes, [configuration](configuration.md) |
| New `PipelineConfig` field | [configuration](configuration.md), [runtime flows](runtime-flows.md) if it affects orchestration |
| GUI option added/removed | [runtime flows](runtime-flows.md), [configuration](configuration.md) |
| Cache schema/stage changes | [data contracts](data-contracts.md), [runtime flows](runtime-flows.md) |
| Debug `.ocr.json` fields changed | [data contracts](data-contracts.md), [debugging](debugging.md) |
| New model backend | `README.md`, [configuration](configuration.md), [architecture](architecture.md), `AGENTS.md` |
| New harness | [autoresearch](autoresearch.md), [testing](testing.md), [wiki-manifest.json](wiki-manifest.json) |
| Test command changes | [testing](testing.md), `README.md` if user-facing |

## Style Rules

- Keep each page scannable with a short `Start Here` section.
- Prefer source anchors and behavior maps over long function inventories.
- Mark local/generated folders clearly.
- Keep examples Windows PowerShell-friendly unless documenting cross-platform harness commands.
- Keep CLI/config tables synchronized with actual code, not memory.

## Verification

After wiki edits:

1. Check Markdown links and referenced paths.
2. Run `python -m manga_local_translator --help` if CLI/config docs changed.
3. Verify `wiki-manifest.json` lists every wiki page.
4. Skim `AGENTS.md` as a first-time agent: it should identify purpose, entrypoints, common tasks, and safety notes quickly.

