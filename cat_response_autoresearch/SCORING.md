# Scoring

Primary quality metric:

```text
cat_approval_rate
```

Higher is better. Target: `>= 0.90`.

Primary optimization score:

```text
cat_quality_score
```

Lower is better.

The primary optimization benchmark is `benchmarks/real_mined`, because it is seeded from real CAT failures and OCR text shapes. `benchmarks/synthetic` is a regression check and harness smoke benchmark. Synthetic-only improvements should not replace the kept best profile.

The kept run score is the average of 4 complete benchmark passes by default. Per-pass metrics are stored in `summary.json` as `repeat_metrics`, and the result row stores averaged counts/rates so random one-off CAT behavior does not become the recorded best.

`cat_response_score` is still reported, but it includes latency. Best-run replacement prioritizes `cat_quality_score`. If approval does not improve, quality score must improve by at least 0.2% before a run replaces best. If approval and quality are tied, lower retry dependence can replace best. If approval, quality, and retry rate all tie, lower response score can replace best as an operational improvement.

Retry diagnostics:

```text
retried_count              = cases where CAT needed the retry prompt
retry_rate                 = retried_count / evaluations_total
primary_rejected_count     = cases where the first CAT output was rejected
retry_rescued_accept_count = expected-good cases accepted only after retry
retry_wasted_safe_reject_count = expected-reject cases retried but still safely rejected
retry_failed_count         = retried cases still not approved
clean_primary_accept_count = expected-good cases accepted from the first CAT output
```

These are not hard failures by themselves. They are tie-breakers and review signals: a 100% run with fewer retries is better than a 100% run that depends on retry rescues.

Outcome classes:

```text
clean_accept  = expected-good line accepted cleanly
safe_reject   = expected-bad/noisy line rejected
false_reject  = expected-good line rejected
unsafe_accept = expected-bad or hard-failure line accepted
weak_accept   = accepted but with non-hard quality warnings
unclear_fail  = any remaining failed shape
```

Category approval is also tracked by `source_type`. A run is not eligible to be kept when any source category with at least two evaluations falls below `0.80` approval, even if the global score improves.

Formula:

```text
cat_quality_score =
  12000 * accepted_prompt_chatter_rate
+ 11000 * accepted_japanese_leakage_rate
+ 10000 * accepted_prompt_fragment_rate
+ 10000 * accepted_schema_fragment_rate
+  9000 * source_text_mutation_rate
+  7000 * accepted_forbidden_pattern_rate
+  7000 * accepted_forbidden_meaning_term_rate
+  6500 * accepted_required_meaning_missing_rate
+  4500 * accepted_overlong_fragment_rate
+  3500 * accepted_repetitive_rate
+  2500 * accepted_explanatory_output_rate
+  1800 * accepted_fragment_shape_warning_rate
+  3000 * false_reject_rate
+  2000 * empty_output_rate
+  1000 * verbose_output_rate
```

`cat_latency_score = 10 * mean_latency_ms`

`cat_response_score = cat_quality_score + cat_latency_score`

Hard failures include accepted prompt chatter, accepted Japanese leakage, accepted prompt/schema fragments, source mutation, malformed fixtures, model crash, invalid JSON artifacts, or benchmark/scoring edits during optimization.

A profile that improves `real_mined` but fails `real_mined_holdout` is treated as overfit. Keep the run artifacts for diagnosis, but do not promote that profile into production CAT behavior.

Category reporting is mandatory. A global score improvement is not enough if one source category gets worse without a clear compensating hard-failure reduction.

Fragment-shape warnings are weak accepts, not hard failures. They catch generic bad shapes such as dangling comma fragments, unbacked apology phrasing on non-apology sources, and romanized SFX where translated SFX is expected.

Semantic term checks:

```text
required_meaning_terms  = accepted outputs must include each required meaning group
forbidden_meaning_terms = accepted outputs must not include any forbidden meaning group
```

Use these only for generic, obvious preservation checks such as honorific names, family terms, crude nouns, and metadata/date traps. Do not encode exact page-specific translations.
