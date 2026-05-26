# Scoring

Primary metric:

```text
source_extraction_score
```

Lower is better.

Diagnostic metrics:

```text
source_extraction_quality_score
timing_score_component
```

`source_extraction_score` is quality-only and equals `source_extraction_quality_score`. `timing_score_component` is the raw p95 extraction milliseconds and is reported as a tie-breaker, not added to the primary score.

Formula:

```text
source_extraction_score =
  12000 * missed_dialogue_region_rate
+ 10000 * destructive_false_positive_rate
+  8000 * wrong_text_region_match_rate
+  7000 * mean_ocr_cer
+  6500 * severe_ocr_error_rate
+  6000 * overmerge_rate
+  6000 * undermerge_rate
+  5000 * reading_order_error_rate
+  4000 * orientation_error_rate
+  3000 * bad_crop_rate
+  2500 * duplicate_region_rate
+  1800 * empty_ocr_rate
+  1500 * non_japanese_noise_rate
+   500 * harmless_false_positive_rate
```

Definitions:

```text
missed_dialogue_region_rate:
  Extractable dialogue/narration/thought/sign/SFX ground-truth regions with no matching prediction, divided by extractable ground-truth regions.

destructive_false_positive_rate:
  Predicted regions that would likely create fake translation or unwanted erase/render, divided by predicted regions.

wrong_text_region_match_rate:
  Matched regions whose OCR text is severely inconsistent with reference text even though the box matched.

mean_ocr_cer:
  Mean capped character error rate over matched extractable regions.

severe_ocr_error_rate:
  Matched extractable regions with capped CER >= 0.50.

overmerge_rate:
  Predicted groups that combine multiple distinct ground-truth groups, divided by predicted groups.

undermerge_rate:
  Ground-truth groups split across multiple predicted groups, divided by ground-truth groups.

reading_order_error_rate:
  Pairwise order inversions over matched comparable groups or regions.

orientation_error_rate:
  Matched extractable regions whose predicted orientation differs from the ground-truth orientation.

bad_crop_rate:
  Matched regions with weak spatial fit, such as IoU < 0.45 and no small-region exception.

duplicate_region_rate:
  Duplicate predicted regions divided by predicted regions.

empty_ocr_rate:
  Matched extractable regions with empty or punctuation-only OCR output.

non_japanese_noise_rate:
  Predicted regions accepted as source text but dominated by non-Japanese noise when the label says they are not extractable.

harmless_false_positive_rate:
  Low-impact false positives, divided by predicted regions.

p95_extraction_ms_per_page:
  95th percentile total source-extraction time per page.
```

Hard failure conditions include failed tests, failed fixture validation, evaluator crashes, missing result rows, edited benchmark/scoring files during optimization, blanked OCR, zero predictions on pages with extractable labels, unstable IDs, invalid JSON outputs, NaN scores, or invocation of translation, rendering, Qwen, CAT, OPUS, MADLAD, Argos, or vision code.

Tie-breakers: lower missed count, lower destructive false positives, lower mean OCR CER, lower severe OCR errors, lower merge errors, lower order errors, lower orientation errors, lower duplicates, lower p95 extraction time, then smaller project-code diff.
