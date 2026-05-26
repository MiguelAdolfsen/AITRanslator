# Keep Experimenting

The user asked Codex to keep running the source extraction autoresearch loop and not stop until explicitly told to stop.

Stay within the program scope:
- source extraction only
- no translation, rendering, erase, vision, CAT, Qwen, OPUS, MADLAD, or Argos calls
- keep benchmark/scoring/results schema frozen
- run tests, fixture validation, and benchmark for each experiment
- keep only changes that improve `source_extraction_score` without violating `PROGRAM.md` guards

Current benchmark guidance:
- `benchmarks/synthetic/` is only a smoke/regression set. It is too clean to be the only source of decisions.
- `benchmarks/cases/local_spy_short_v1_frozen/` is the current local real-page benchmark from the Spy x Family short chapter.
- Use the frozen real case for keep/revert decisions after synthetic passes.
- `benchmarks/cases/local_spy_short_v1_draft/` remains only as scaffolding/reference and should not drive decisions.
- The frozen case policy is core extraction first: dialogue, narration, signs, thought bubbles, and tight speech-like SFX are scored; credits, promo/sidebar text, footer/date text, punctuation-only bubbles, and large decorative SFX are ignored or non-extractable.

Suggested local real-case benchmark command:

```powershell
.\.venv\Scripts\python.exe source_extraction_autoresearch\scripts\eval_source_extraction.py `
  --benchmark source_extraction_autoresearch\benchmarks\cases\local_spy_short_v1_frozen `
  --output source_extraction_autoresearch\runs\local_spy_current `
  --results source_extraction_autoresearch\results\results.tsv `
  --run-id local_spy_current `
  --tests-ok `
  --overwrite-output
```
