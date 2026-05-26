# Benchmarks

`synthetic/` contains deterministic generated pages for harness smoke tests and regression checks.
`cases/` is reserved for real hand-labeled page sets.
`fixtures/` is reserved for compact shared fixture data.

Synthetic fixtures are intentionally cheap and deterministic, but they are too clean to be the only optimization target.
Real page sets under `cases/` should drive source-extraction decisions once their labels are manually reviewed.

Local real manga pages should use a `local_` prefix, for example:

```text
source_extraction_autoresearch/benchmarks/cases/local_real_diverse_v2_frozen/
```

`local_*` case folders are ignored by git because they can contain copyrighted page images. They are still valid local benchmarks for this machine.

Draft labels created from `.ocr.json` are not frozen ground truth. They must be manually checked and copied into a frozen case before an autoresearch agent uses them as a keep/revert benchmark. Draft case folders should not live in `benchmarks/cases/` once a frozen replacement exists.

Current local case policy:

- `local_real_diverse_v2_frozen/` is the primary broad real-page decision benchmark.
- `local_spy_short_v1_frozen/` and `local_hard_pages_v1_frozen/` are historical v1 real-page regression sets.
- Draft cases and empty scaffold labels must never drive keep/revert decisions.

The v2 frozen case should target broad source-extraction coverage: vertical, horizontal, and mixed text; dialogue, narration, signs, thoughts, and tight SFX; non-extractable credits, metadata, page numbers, noise, punctuation-only bubbles, and decorative SFX; split groups; close-but-separate bubbles; and hard OCR pages with low contrast, dense art, tiny text, edge text, and text touching art lines.
