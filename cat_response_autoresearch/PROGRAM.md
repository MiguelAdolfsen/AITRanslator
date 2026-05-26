# CAT Response Autoresearch Program

Objective: maximize `cat_approval_rate` and minimize `cat_quality_score`.

Target: `cat_approval_rate >= 0.90` with zero hard failures.

Scope: CAT prompt profiles, deterministic CAT generation settings, output cleanup, output validation, and retry prompt policy only.

Frozen benchmark rule: do not modify benchmark cases, references, scoring, result schema, templates, or historical result rows during optimization.

Anti-overfit rule: do not tune for exact source strings, exact chapter names, page IDs, folder names, or known benchmark lines. Kept rules must be explainable by generic source shape or output behavior: short fragment, ellipsis, punctuation-heavy line, SFX, katakana/name-like source, OCR noise, metadata-like source, prompt chatter, Japanese leakage, repeated output, or overlong output.

Required validation:

```powershell
.\.venv\Scripts\python.exe cat_response_autoresearch\scripts\validate_fixtures.py `
  --benchmark cat_response_autoresearch\benchmarks\synthetic
```

Required unit tests:

```powershell
.\.testing\run_tests.ps1
```

Required benchmark:

```powershell
.\.venv\Scripts\python.exe cat_response_autoresearch\scripts\eval_cat_responses.py `
  --benchmark cat_response_autoresearch\benchmarks\synthetic `
  --output cat_response_autoresearch\runs\current `
  --results cat_response_autoresearch\results\results.tsv `
  --run-id current
```

The required benchmark uses the evaluator default of `--repeats 4`. Keep/best decisions must be based on the averaged four-pass result, not a single lucky CAT run. `--repeats 1` is allowed only for local smoke checks that are not written as best.

Noise guardrails:

```text
- Best-run replacement uses cat_quality_score, not latency-included cat_response_score.
- cat_response_score is still reported for visibility, but latency cannot make a run best by itself.
- If approval does not improve, cat_quality_score must improve by at least 1% before a run can replace best.
- For visibly noisy CAT behavior, rerun likely keepers with --repeats 6 or --repeats 8 before holdout.
```

Required holdout check after a kept main improvement:

```powershell
.\.venv\Scripts\python.exe cat_response_autoresearch\scripts\eval_cat_responses.py `
  --benchmark cat_response_autoresearch\benchmarks\holdout `
  --output cat_response_autoresearch\runs\holdout_current `
  --results cat_response_autoresearch\results\results.tsv `
  --run-id holdout_current `
  --no-update-best
```

The holdout is not an optimization target. Use it to catch overfit prompt/settings behavior before promoting a profile.

Holdout cadence:

```text
- Run the main synthetic benchmark for every experiment.
- If the main run is not kept, do not run holdout.
- If the main run is kept or would replace current best, immediately run holdout with the same profile/settings.
- Do not tune against holdout failures directly. If holdout exposes a generic category weakness, add new generic training cases to the main benchmark in a separate review pass.
- A profile cannot be promoted into production CAT behavior unless main + holdout both have zero hard failures.
```

Editable project-code surface after harness setup:

```text
manga_local_translator/hf_translators.py
manga_local_translator/translation_rules.py
```

Forbidden edits during optimization:

```text
cat_response_autoresearch/benchmarks/**
cat_response_autoresearch/scripts/eval_cat_responses.py
cat_response_autoresearch/scripts/score.py
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
.testing/tests/**
```

Keep a change only if:

```text
1. unit tests pass,
2. fixture validation passes,
3. cat_quality_score improves versus current best by the required margin on the averaged four-pass benchmark, or cat_approval_rate improves,
4. cat_approval_rate does not decrease,
5. accepted_prompt_chatter_count remains 0,
6. accepted_japanese_leakage_count remains 0,
7. accepted_prompt_fragment_count remains 0,
8. accepted_schema_fragment_count remains 0,
9. no source category with at least two evaluations falls below 0.80 approval,
10. false_reject_count does not increase unless hard failures decrease,
11. the holdout check does not introduce hard failures,
12. the improvement did not come from editing benchmark/scoring/logging files or exact frozen strings.
```

Otherwise revert the experiment and append a failed result row.
