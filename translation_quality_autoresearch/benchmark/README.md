# Benchmark

Seed cases are synthetic, short Japanese dialogue fragments created for harness validation. They are not copied manga text.

`cases.jsonl` is the default replay benchmark. Additional subset files may be passed as a comma-separated list to `--cases`.

The current checked-in cases are a seed/smoke benchmark. They are useful for proving the harness works, but they are not enough for autonomous optimization. Run:

```bash
python translation_quality_autoresearch/scripts/report_benchmark_coverage.py \
  --cases translation_quality_autoresearch/benchmark/cases.jsonl,translation_quality_autoresearch/benchmark/adversarial_cases.jsonl,translation_quality_autoresearch/benchmark/page_context_cases.jsonl
```

See `VERSIONING.md` before creating a larger reviewed benchmark.
