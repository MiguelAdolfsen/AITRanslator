# CAT Response Autoresearch

This folder is an isolated research loop for CAT-Translate response reliability.

It benchmarks live CAT output over frozen OCR source-text cases. It does not run OCR, detection, rendering, erase, Qwen, Q8, OPUS, MADLAD, Argos, vision, GUI code, or model downloads.

The goal is to make CAT-alone responses reliable enough for production experiments, with a target of `cat_approval_rate >= 0.90`.

## Quick Commands

Dashboard:

```powershell
.\.venv\Scripts\python.exe cat_response_autoresearch\scripts\serve_dashboard.py
```

Then open `http://127.0.0.1:8501`. The dashboard polls `results/results.tsv`, `results/best.json`, and `runs/*/progress.json` every two seconds.

```powershell
.\.venv\Scripts\python.exe cat_response_autoresearch\scripts\validate_fixtures.py `
  --benchmark cat_response_autoresearch\benchmarks\synthetic

.\.venv\Scripts\python.exe cat_response_autoresearch\scripts\eval_cat_responses.py `
  --benchmark cat_response_autoresearch\benchmarks\synthetic `
  --output cat_response_autoresearch\runs\current `
  --results cat_response_autoresearch\results\results.tsv `
  --run-id current
```

The evaluator runs the full benchmark 4 times by default and records the average score. Use `--repeats 1` only for quick harness debugging, not for keeping a profile as best.

Best-run updates use `cat_quality_score`, not latency-included `cat_response_score`. If approval rate does not improve, a run must improve quality score by at least 1% to replace best. For noisy experiments, rerun likely keepers with `--repeats 6` or `--repeats 8` before the holdout check.

After a profile is kept on the main benchmark, run the locked holdout as a sanity check:

```powershell
.\.venv\Scripts\python.exe cat_response_autoresearch\scripts\eval_cat_responses.py `
  --benchmark cat_response_autoresearch\benchmarks\holdout `
  --output cat_response_autoresearch\runs\holdout_current `
  --results cat_response_autoresearch\results\results.tsv `
  --run-id holdout_current `
  --no-update-best
```

Do not run holdout for every failed experiment. It is a confirmation step for a kept/improved profile, and it should not become the optimization target.

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
