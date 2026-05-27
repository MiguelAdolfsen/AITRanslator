# CAT Response Autoresearch Program

Objective: maximize `cat_approval_rate` and minimize `cat_quality_score`.

Target: `cat_approval_rate >= 0.90` with zero hard failures.

Scope: deterministic CAT generation settings, output cleanup, output validation, retry acceptance, and generic source-shape safeguards only.

Prompt freeze rule: the CAT production prompt is frozen to the strict fragment prompt used by the saved strict benchmark. The normal retry prompt is the source-only `Japanese:\n...\n\nEnglish:` form; the incomplete-fragment prompt is only for production second retry. Do not edit `build_cat_prompt`, `CAT_PROMPT_VERSION`, retry prompt constants, built-in prompt templates, system prompt text, or profile prompt JSON to improve benchmark results. Prompt changes require explicit user approval outside this autoresearch loop.

Frozen benchmark rule: do not modify benchmark cases, references, scoring, result schema, templates, or historical result rows during optimization.

Anti-overfit rule: do not tune for exact source strings, exact chapter names, page IDs, folder names, or known benchmark lines. Kept rules must be explainable by generic source shape or output behavior: short fragment, ellipsis, punctuation-heavy line, SFX, katakana/name-like source, OCR noise, metadata-like source, prompt chatter, Japanese leakage, repeated output, or overlong output.

Required validation:

```powershell
.\.venv\Scripts\python.exe cat_response_autoresearch\scripts\validate_fixtures.py `
  --benchmark cat_response_autoresearch\benchmarks\real_mined

.\.venv\Scripts\python.exe cat_response_autoresearch\scripts\validate_fixtures.py `
  --benchmark cat_response_autoresearch\benchmarks\synthetic

.\.venv\Scripts\python.exe cat_response_autoresearch\scripts\validate_fixtures.py `
  --benchmark cat_response_autoresearch\benchmarks\real_mined_holdout
```

Required unit tests:

```powershell
.\.testing\run_tests.ps1
```

Main optimization benchmark:

```powershell
.\.venv\Scripts\python.exe cat_response_autoresearch\scripts\eval_cat_responses.py `
  --benchmark cat_response_autoresearch\benchmarks\real_mined `
  --output cat_response_autoresearch\runs\real_mined_current `
  --results cat_response_autoresearch\results\results.tsv `
  --run-id real_mined_current `
  --profile production
```

The main benchmark uses the evaluator default of `--repeats 4`. Keep/best decisions must be based on the averaged four-pass `real_mined` result, not a single lucky CAT run. `--repeats 1` is allowed only for local smoke checks that are not written as best.

Synthetic regression check:

```powershell
.\.venv\Scripts\python.exe cat_response_autoresearch\scripts\eval_cat_responses.py `
  --benchmark cat_response_autoresearch\benchmarks\synthetic `
  --output cat_response_autoresearch\runs\synthetic_regression_current `
  --results cat_response_autoresearch\results\results.tsv `
  --run-id synthetic_regression_current `
  --no-update-best
```

Synthetic is a regression guard and harness smoke benchmark. It is not the main keep gate because it is cleaner than real OCR output and can make overfit prompt/validation changes look better than they are.

Use `--profile production` when checking whether the harness matches current production CAT behavior. It calls production prompt construction and token-cap settings directly. The dashboard may display the production prompt text, but agents must not modify it.

The evaluator resolves the CAT GGUF from the project root `.models/CAT-Translate` directory, even if the command is launched from inside `cat_response_autoresearch`. Do not work around model lookup by copying model files into benchmark folders.

Blind reporting mode:

```powershell
.\.venv\Scripts\python.exe cat_response_autoresearch\scripts\eval_cat_responses.py `
  --benchmark cat_response_autoresearch\benchmarks\real_mined `
  --output cat_response_autoresearch\runs\real_mined_blind_current `
  --results cat_response_autoresearch\results\results.tsv `
  --run-id real_mined_blind_current `
  --profile production `
  --blind-report
```

Use blind reporting when an optimization agent should receive benchmark feedback without seeing the frozen source phrases or CAT outputs. Blind artifacts may include case IDs, source category, source shape, risk labels, violation labels, retry status, and aggregate metrics. They must not include `source_text`, raw CAT output, final CAT output, exact references, required meaning terms, forbidden meaning terms, chapter titles, or page-specific text. Owner-only unredacted artifacts may be written with `--private-output`, but that path should be outside the agent workspace.

Private blind benchmark wrapper:

```powershell
$env:CAT_PRIVATE_BENCHMARK = "C:\Users\migue\Desktop\private_manga_benchmarks\cat_real_mined_blind"
.\cat_response_autoresearch\scripts\run_private_blind_benchmark.ps1 `
  -RunId private_blind_current `
  -Profile production `
  -NoUpdateBest
```

Use this wrapper when the benchmark JSONL should remain hidden from an optimization agent. It refuses benchmark paths inside the repo, checks for `cases.jsonl` and `references.jsonl`, validates fixtures, and always runs the evaluator with `--blind-report`. If `$env:CAT_PRIVATE_OUTPUT` is set, it must also point outside the repo; otherwise the wrapper refuses to run.

Security model:

```text
- .gitignore is not access control.
- Soft blind means private files are outside the repo and only sanitized artifacts are written into the repo.
- Hard blind requires an OS/account/process boundary where the agent cannot read the private files at all.
- Do not give an agent direct benchmark paths or unredacted private output folders.
```

Noise guardrails:

```text
- Best-run replacement prioritizes cat_quality_score, not latency-included cat_response_score.
- cat_response_score is still reported for visibility. It can only replace best as a final tie-breaker when approval, quality, and retry_rate are all tied.
- If approval does not improve, cat_quality_score must improve by at least 0.2% before a run can replace best.
- If approval and quality are tied, lower retry dependence can replace best. This keeps useful prompt/settings improvements visible even after quality reaches 0.
- For visibly noisy CAT behavior, rerun likely keepers with --repeats 6 or --repeats 8 before holdout.
- If best.json is missing or stale, the evaluator compares against the last kept result row for the same benchmark_set in results.tsv.
```

Required holdout check after a kept main improvement:

```powershell
.\.venv\Scripts\python.exe cat_response_autoresearch\scripts\eval_cat_responses.py `
  --benchmark cat_response_autoresearch\benchmarks\real_mined_holdout `
  --output cat_response_autoresearch\runs\real_mined_holdout_current `
  --results cat_response_autoresearch\results\results.tsv `
  --run-id real_mined_holdout_current `
  --profile production `
  --no-update-best
```

The holdout is not an optimization target. Use it to catch overfit prompt/settings behavior before promoting a profile. A profile that wins `real_mined` but fails holdout must not be promoted; record it as an overfit experiment instead.

Holdout cadence:

```text
- Run `real_mined` as the main benchmark for CAT response quality experiments.
- Run `synthetic` as a regression check with `--no-update-best`; do not tune to synthetic-only gains.
- If the `real_mined` run is not kept, do not run holdout.
- If the `real_mined` run is kept or would replace current best, immediately run `real_mined_holdout` with the same profile/settings.
- Do not tune against holdout failures directly. If holdout exposes a generic category weakness, add new generic training cases to the main benchmark in a separate review pass.
- A profile cannot be promoted into production CAT behavior unless synthetic regression, real_mined, and real_mined_holdout all have zero hard failures.
```

Editable project-code surface after harness setup:

```text
manga_local_translator/hf_translators.py
manga_local_translator/translation_rules.py
```

When editing `manga_local_translator/hf_translators.py`, only cleanup, validation, retry acceptance, and source-shape guard code is in scope. Prompt builders and prompt constants are frozen.

Phrasebook rule:

```text
manga_local_translator/translation_phrasebook.py is not part of CAT response optimization.
Do not edit it from autoresearch. Phrasebook additions require separate manual review and must be broad manga expressions/SFX, not mined benchmark lines, names, or plot terms.
```

Forbidden edits during optimization:

```text
cat_response_autoresearch/benchmarks/**
cat_response_autoresearch/scripts/eval_cat_responses.py
cat_response_autoresearch/scripts/score.py
cat_response_autoresearch/scripts/io_adapters.py prompt/profile definitions
cat_response_autoresearch/scripts/validate_fixtures.py
cat_response_autoresearch/results/results.tsv
translation_routing_autoresearch/**
source_extraction_autoresearch/**
render_autoresearch/**
manga_local_translator/render.py
manga_local_translator/erase.py
manga_local_translator/detect_ocr.py
manga_local_translator/grouping.py
manga_local_translator/qwen_translator.py
manga_local_translator/qwen_validation.py
manga_local_translator/translation_phrasebook.py
.testing/tests/**
```

Keep a change only if:

```text
1. unit tests pass,
2. fixture validation passes,
3. cat_quality_score improves versus current best by the required margin on the averaged four-pass `real_mined` benchmark, cat_approval_rate improves, approval/quality tie while retry_rate improves, or approval/quality/retry tie while response score improves,
4. cat_approval_rate does not decrease,
5. accepted_prompt_chatter_count remains 0,
6. accepted_japanese_leakage_count remains 0,
7. accepted_prompt_fragment_count remains 0,
8. accepted_schema_fragment_count remains 0,
9. no source category with at least two evaluations falls below 0.80 approval,
10. false_reject_count does not increase unless hard failures decrease,
11. synthetic regression and real_mined_holdout do not introduce hard failures,
12. the improvement did not come from editing benchmark/scoring/logging files or exact frozen strings.

Retry-dependent runs can be kept for iteration, but they are not automatically production-ready. Prefer profiles that reduce `retried_count`, `retry_rate`, and `primary_rejected_count` without introducing false rejects or weak accepts.

Production promotion rule:

```text
Before promoting a CAT profile/settings change into manga_local_translator, run synthetic, real_mined, and the matching holdout. The promoted change must have zero hard failures, zero accepted semantic trap failures, and no regression in retry dependence unless it fixes a hard failure.
```
```

Otherwise revert the experiment and append a failed result row.
