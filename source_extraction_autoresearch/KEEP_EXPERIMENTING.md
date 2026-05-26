# Keep Experimenting

The user asked Codex to keep running the source extraction autoresearch loop and not stop until explicitly told to stop.

Stay within the program scope:
- source extraction only
- no translation, rendering, erase, vision, CAT, Qwen, OPUS, MADLAD, or Argos calls
- keep benchmark/scoring/results schema frozen
- run the matrix gate or run evaluator-owned tests with `--run-tests`; do not rely on old trust-based test flags
- keep only changes that improve quality-only `source_extraction_score` without violating `PROGRAM.md` guards

Current benchmark guidance:
- `benchmarks/synthetic/` is only a smoke/regression set. It is too clean to be the only source of decisions.
- `benchmarks/cases/local_real_diverse_v2_frozen/` is the primary broad real-page benchmark. Use it after synthetic passes.
- `benchmarks/cases/local_spy_short_v1_frozen/` is the historical v1 Spy x Family real-page regression benchmark.
- `benchmarks/cases/local_hard_pages_v1_frozen/` is the historical v1 hard-page real benchmark sampled from Kaguya, Paladin, and Frieren.
- Use a matrix gate for keep/revert decisions after synthetic passes: v2 frozen is required; v1 Spy and v1 hard pages are historical regression checks for broad behavior changes.
- Draft case folders and empty scaffold labels must not drive decisions and should not remain under `benchmarks/cases/` once a frozen replacement exists.
- The frozen case policy is core extraction first: dialogue, narration, signs, thought bubbles, and tight speech-like SFX are scored; credits, promo/sidebar text, footer/date text, punctuation-only bubbles, and large decorative SFX are ignored or non-extractable.
- `source_extraction_score` is quality-only. `p95_extraction_ms_per_page` and `timing_score_component` are tie-breakers, not primary score components.
- Decision runs are strict: no adapter-smoke, no draft benchmarks, no skipped validation, no dirty forbidden files, and tests must be run by the harness or supplied by the matrix test context.

Only create a v2 draft scaffold when adding new real pages. After those pages are manually reviewed and copied into a frozen case, remove the draft folder from `benchmarks/cases/`:

```powershell
.\.venv\Scripts\python.exe source_extraction_autoresearch\scripts\build_real_case_inventory.py `
  --source-root mangafolder `
  --output source_extraction_autoresearch\benchmarks `
  --per-folder 6
```

Preferred matrix gate command:

```powershell
.\.venv\Scripts\python.exe source_extraction_autoresearch\scripts\run_benchmark_matrix.py `
  --include-historical `
  --overwrite-output
```

Suggested primary v2 real-case benchmark command when not using the matrix:

```powershell
.\.venv\Scripts\python.exe source_extraction_autoresearch\scripts\eval_source_extraction.py `
  --benchmark source_extraction_autoresearch\benchmarks\cases\local_real_diverse_v2_frozen `
  --output source_extraction_autoresearch\runs\local_real_diverse_v2_current `
  --results source_extraction_autoresearch\results\results.tsv `
  --run-id local_real_diverse_v2_current `
  --run-tests `
  --overwrite-output
```

Suggested historical v1 Spy regression command:

```powershell
.\.venv\Scripts\python.exe source_extraction_autoresearch\scripts\eval_source_extraction.py `
  --benchmark source_extraction_autoresearch\benchmarks\cases\local_spy_short_v1_frozen `
  --output source_extraction_autoresearch\runs\local_spy_current `
  --results source_extraction_autoresearch\results\results.tsv `
  --run-id local_spy_current `
  --run-tests `
  --overwrite-output
```

Suggested historical v1 hard-page regression command:

```powershell
.\.venv\Scripts\python.exe source_extraction_autoresearch\scripts\eval_source_extraction.py `
  --benchmark source_extraction_autoresearch\benchmarks\cases\local_hard_pages_v1_frozen `
  --output source_extraction_autoresearch\runs\local_hard_pages_current `
  --results source_extraction_autoresearch\results\results.tsv `
  --run-id local_hard_pages_current `
  --run-tests `
  --overwrite-output
```
