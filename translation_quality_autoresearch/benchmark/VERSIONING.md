# Benchmark Versioning

The checked-in `benchmark/` folder is the seed/smoke benchmark. It verifies that the harness, deterministic checks, reranking, result logs, and replay evaluation work.

Do not use the seed benchmark alone for autonomous prompt or rule optimization. It is too small and synthetic to represent the full project.

## Recommended versions

- `benchmark/`: seed/smoke benchmark for fast harness checks.
- `benchmark_v1/`: first reviewed real benchmark, 60-100 cases minimum.
- `benchmark_holdout_v1/`: reviewed holdout set, 20-30% of the v1 case count.
- `results/translation_quality_results_v1.tsv`: result log for v1 only.

Never compare scores from different benchmark versions without naming the version.

## Case acceptance rules

Generated cases are candidates for review, not trusted gold. A case is ready for a real benchmark only when it has:

- Stable `case_id` and `source_hash`.
- Source text copied from validated OCR/debug output.
- Context fields when available.
- Tags describing the failure mode or scenario.
- At least one reference translation.
- Known-bad translations or forbidden additions for likely failures.
- Frozen outputs or a documented live-trace source.
- Human-gold constraints when the case represents a critical regression.

## Minimum v1 coverage target

- 20 clean dialogue cases.
- 10 short fragment or reaction cases.
- 8 SFX cases.
- 10 ambiguous speaker, subject, or context cases.
- 8 names, honorifics, or glossary cases.
- 10 noisy but salvageable OCR cases.
- 8 hallucination traps.
- 8 page-context-required cases.
- 5 multi-line or line-mapping cases.
- 5 oververbose or renderability cases.

These are minimums, not caps. Prefer real project failures over synthetic examples once the harness workflow is stable.
