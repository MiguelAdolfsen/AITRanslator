# Benchmark Coverage Report

- readiness: NEEDS_REVIEWED_BENCHMARK_EXPANSION
- total_cases: 50
- with_references: 50
- with_human_gold: 0
- with_context: 50
- with_glossary_terms: 0
- with_must_preserve: 0
- multi_line_source: 0

## Source Types
- dialogue: 41
- reaction: 8
- sfx: 1

## OCR Risks
- clean: 42
- noisy_salvageable: 8

## Tags
- agent_rejected_candidate: 8
- ambiguous_subject: 5
- ellipsis: 9
- generated: 50
- names_and_honorifics: 8
- needs_human_review: 50
- ocr_noisy_but_salvageable: 8
- page_context_available: 50
- reaction: 9
- sfx: 1
- short_fragment: 19

## Readiness Gaps
- total_cases: actual=50 target=60 missing=10
- sfx: actual=1 target=8 missing=7
- reaction: actual=9 target=10 missing=1
- ambiguous_speaker: actual=0 target=5 missing=5
- glossary_sensitive: actual=0 target=8 missing=8
- ocr_noisy_but_salvageable: actual=8 target=10 missing=2
- hallucination_trap: actual=0 target=8 missing=8
- page_context_required: actual=0 target=8 missing=8
- line_mapping: actual=0 target=5 missing=5
- oververbose_guard: actual=0 target=5 missing=5
- human_gold: actual=0 target=10 missing=10
