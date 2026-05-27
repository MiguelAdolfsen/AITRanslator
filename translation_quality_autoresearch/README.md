# Translation Quality Autoresearch

This folder is for the isolated translation-quality harness and its benchmark fixtures.

## Vision-Facts Harness

Validate the current benchmark:

```powershell
.\.venv\Scripts\python.exe translation_quality_autoresearch\scripts\validate_benchmark.py `
  --benchmark translation_quality_autoresearch\benchmarks\frieren_ch26_pages_002_010
```

Smoke-test the harness without loading CAT/Qwen/vision models:

```powershell
.\.venv\Scripts\python.exe translation_quality_autoresearch\scripts\run_vision_facts_eval.py `
  --fake `
  --limit-pages frieren_ch26_002 `
  --run-id fake_smoke
```

Run a live paired eval against the current best profile:

```powershell
.\.venv\Scripts\python.exe translation_quality_autoresearch\scripts\run_vision_facts_eval.py `
  --candidate-profile current_production `
  --run-id current_production_live
```

Add `--promote` only when the candidate profile should be committed and pushed automatically after passing the main and holdout gates. The harness never edits benchmark labels as part of promotion.

## Frieren Reference Labeler

Use the GUI helper to label the English same-layout Frieren pages against the frozen Japanese prepared-page caches:

```powershell
.\.venv\Scripts\python.exe translation_quality_autoresearch\label_frieren_references.py
```

The default setup uses:

- Japanese pages: `mangafolder\sousou-no-frieren-chapter-26`
- English pages: `mangafolder\Frieren-Chapter 26 Present for a Warrior-ENGLISH`
- Prepared caches: `quality-runs\frieren-cat-final-20260527\quality\.manga-work`
- Pages: `2` through `10`
- Output: `translation_quality_autoresearch\benchmarks\frieren_ch26_pages_002_010`

Label each Japanese line by selecting the matching English bubble in the same page location and transcribing the English text. Use the official English as a semantic reference, not as exact wording the translator must copy. Mark lines as skipped when the English page does not provide a usable corresponding translation.
