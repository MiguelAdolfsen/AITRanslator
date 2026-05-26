# CAT Response Autoresearch

This folder is an isolated research loop for CAT-Translate response reliability.

It benchmarks live CAT output over frozen OCR source-text cases. It does not run OCR, detection, rendering, erase, Qwen, Q8, OPUS, MADLAD, Argos, vision, GUI code, or model downloads.

The goal is to make CAT-alone responses reliable enough for production experiments, with a target of `cat_approval_rate >= 0.90`.

## Quick Commands

Dashboard:

```powershell
.\.venv\Scripts\python.exe cat_response_autoresearch\scripts\serve_dashboard.py
```

Then open `http://127.0.0.1:8501`. The dashboard polls `results/results.tsv`, `results/best.json`, and `runs/*/progress.json` every two seconds. It labels `real_mined` as the main benchmark, `synthetic` as regression, and holdout runs as confirmation checks.

```powershell
.\.venv\Scripts\python.exe cat_response_autoresearch\scripts\validate_fixtures.py `
  --benchmark cat_response_autoresearch\benchmarks\real_mined

.\.venv\Scripts\python.exe cat_response_autoresearch\scripts\eval_cat_responses.py `
  --benchmark cat_response_autoresearch\benchmarks\real_mined `
  --output cat_response_autoresearch\runs\real_mined_current `
  --results cat_response_autoresearch\results\results.tsv `
  --run-id real_mined_current `
  --profile production
```

These commands can be launched from the repo root or from inside `cat_response_autoresearch`; the harness resolves the local CAT GGUF from the project root `.models/CAT-Translate` directory.

The evaluator runs the full benchmark 4 times by default and records the average score. Use `--repeats 1` only for quick harness debugging, not for keeping a profile as best.

Best-run updates are based on the main `real_mined` benchmark. They prioritize `cat_quality_score`, not latency-included `cat_response_score`. If `best.json` is missing or stale, the evaluator compares against the last kept row in `results.tsv` for the same benchmark. If approval rate does not improve, a run must improve quality score by at least 0.2% to replace best. If quality and approval are tied, a run can still replace best when it reduces retry dependence. If quality, approval, and retry rate all tie, a lower response score can be kept as an operational improvement. For noisy experiments, rerun likely keepers with `--repeats 6` or `--repeats 8` before the holdout check.

Synthetic is a regression check, not the main optimization target. Run it with `--no-update-best` before keeping or promoting a profile:

```powershell
.\.venv\Scripts\python.exe cat_response_autoresearch\scripts\eval_cat_responses.py `
  --benchmark cat_response_autoresearch\benchmarks\synthetic `
  --output cat_response_autoresearch\runs\synthetic_regression_current `
  --results cat_response_autoresearch\results\results.tsv `
  --run-id synthetic_regression_current `
  --no-update-best
```

The real-mined benchmark targets failures found in full manga runs: honorific/name drift, short fragments, semantic traps, and OCR metadata/noise. It is intentionally harder and more representative than the clean synthetic fixture.

After a profile is kept on `real_mined`, run `benchmarks\real_mined_holdout` with `--no-update-best`. Do not tune directly against the holdout, and do not promote a profile that fails holdout:

```powershell
.\.venv\Scripts\python.exe cat_response_autoresearch\scripts\eval_cat_responses.py `
  --benchmark cat_response_autoresearch\benchmarks\real_mined_holdout `
  --output cat_response_autoresearch\runs\real_mined_holdout_current `
  --results cat_response_autoresearch\results\results.tsv `
  --run-id real_mined_holdout_current `
  --profile production `
  --no-update-best
```

For a harness smoke test without loading CAT:

```powershell
.\.venv\Scripts\python.exe cat_response_autoresearch\scripts\eval_cat_responses.py `
  --benchmark cat_response_autoresearch\benchmarks\synthetic `
  --output cat_response_autoresearch\runs\fake_smoke `
  --results cat_response_autoresearch\results\results.tsv `
  --run-id fake_smoke `
  --fake-outputs cat_response_autoresearch\benchmarks\synthetic\fake_outputs.jsonl `
  --repeats 1 `
  --no-update-best
```

## Boundaries

This loop may experiment with CAT prompt profiles, bounded output settings, retry prompt shape, and CAT cleanup/validation. It must not use Qwen/Q8 as backup and must not make decisions from exact frozen-page strings.

Each run writes `comparison.md` next to `human_review.md` to summarize deltas against the current best run. Keep decisions should still inspect failures and per-category metrics, not only the global score.

`human_review.md` also reports retry dependence. A 100% approval run is stronger when most accepted lines are clean primary CAT outputs instead of retry rescues.

Real-mined references may include `required_meaning_terms` and `forbidden_meaning_terms`. These are generic semantic guards, not exact expected translations.
