# Benchmarks

`synthetic/` contains deterministic generated pages for harness smoke tests and regression checks.
`cases/` is reserved for real hand-labeled page sets.
`fixtures/` is reserved for compact shared fixture data.

Synthetic fixtures are intentionally cheap and deterministic, but they are too clean to be the only optimization target.
Real page sets under `cases/` should drive source-extraction decisions once their labels are manually reviewed.

Local real manga pages should use a `local_` prefix, for example:

```text
source_extraction_autoresearch/benchmarks/cases/local_spy_short_v1_draft/
```

`local_*` case folders are ignored by git because they can contain copyrighted page images. They are still valid local benchmarks for this machine.

Draft labels created from `.ocr.json` are not frozen ground truth. They must be manually checked before an autoresearch agent uses the case as a keep/revert benchmark.
