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

The kept run score is the average of 4 complete benchmark passes by default. Per-pass metrics are stored in `summary.json` as `repeat_metrics`, and the result row stores averaged counts/rates so random one-off CAT behavior does not become the recorded best.

`cat_response_score` is still reported, but it includes latency. Best-run replacement uses `cat_quality_score` only. If approval does not improve, quality score must improve by at least 1% before a run replaces best.

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
+  4500 * accepted_overlong_fragment_rate
+  3500 * accepted_repetitive_rate
+  3000 * false_reject_rate
+  2000 * empty_output_rate
+  1000 * verbose_output_rate
```

`cat_latency_score = 10 * mean_latency_ms`

`cat_response_score = cat_quality_score + cat_latency_score`

Hard failures include accepted prompt chatter, accepted Japanese leakage, accepted prompt/schema fragments, source mutation, malformed fixtures, model crash, invalid JSON artifacts, or benchmark/scoring edits during optimization.

Category reporting is mandatory. A global score improvement is not enough if one source category gets worse without a clear compensating hard-failure reduction.
