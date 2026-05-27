# Generated Benchmark Drafts

This folder contains benchmark case drafts mined from local `quality-runs/` artifacts.

This folder is provenance, not a canonical benchmark. Use `translation_quality_autoresearch/benchmark_v1/` for machine-assisted dev scoring and `translation_quality_autoresearch/benchmark_holdout_v1/` for final gating.

## Files

- `quality_runs_50_cases.jsonl`: 50 generated `TranslationCase` records.
- `quality_runs_50_references.todo.jsonl`: human-review template for references.
- `quality_runs_50_references.reviewed.jsonl`: machine-assisted English references.
- `quality_runs_50_coverage.md`: coverage report for the 50 generated cases.
- `quality_runs_50_reviewed_coverage.md`: coverage report using the reviewed references.

## Review status

These cases are not ready for scoring yet. They are marked with:

```json
"metadata": {
  "generated_from_quality_run": true,
  "needs_human_review": true
}
```

Machine translations from the source quality run are stored as `metadata.draft_machine_translation` for reviewer context only. They must not be treated as gold references without review.

`quality_runs_50_references.reviewed.jsonl` contains machine-assisted English references produced from the Japanese source text and context. Some OCR-heavy cases are marked with `review_status: machine_assisted_reference_needs_native_review`; those should not become human-gold gates until a Japanese reader checks them against the page image.

## Next steps

1. Review `quality_runs_50_references.reviewed.jsonl`, especially rows marked `machine_assisted_reference_needs_native_review`.
2. Add or tighten known-bad translations and forbidden additions for high-risk cases.
3. Promote reviewed records into an explicit benchmark version, for example `benchmark_v1/`.
4. Generate or collect frozen outputs for the same case IDs.
5. Run `report_benchmark_coverage.py` against the promoted benchmark.
