# Benchmark Coverage Report

- readiness: NEEDS_REVIEWED_BENCHMARK_EXPANSION
- total_cases: 50
- with_references: 0
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

## Missing References
- generated_quality_00001
- generated_quality_00002
- generated_quality_00003
- generated_quality_00004
- generated_quality_00005
- generated_quality_00006
- generated_quality_00007
- generated_quality_00008
- generated_quality_00009
- generated_quality_00010
- generated_quality_00011
- generated_quality_00012
- generated_quality_00013
- generated_quality_00014
- generated_quality_00015
- generated_quality_00016
- generated_quality_00017
- generated_quality_00018
- generated_quality_00019
- generated_quality_00020
- generated_quality_00021
- generated_quality_00022
- generated_quality_00023
- generated_quality_00024
- generated_quality_00025
- generated_quality_00026
- generated_quality_00027
- generated_quality_00028
- generated_quality_00029
- generated_quality_00030
- generated_quality_00031
- generated_quality_00032
- generated_quality_00033
- generated_quality_00034
- generated_quality_00035
- generated_quality_00036
- generated_quality_00037
- generated_quality_00038
- generated_quality_00039
- generated_quality_00040
- generated_quality_00041
- generated_quality_00042
- generated_quality_00043
- generated_quality_00044
- generated_quality_00045
- generated_quality_00046
- generated_quality_00047
- generated_quality_00048
- generated_quality_00049
- generated_quality_00050

## References Outside Selected Cases
- adv_forbidden_label_001
- page_context_reply_001
- seed_ambiguous_subject_004
- seed_assistant_chatter_010
- seed_clean_dialogue_001
- seed_ellipsis_fragment_002
- seed_glossary_006
- seed_hallucination_trap_009
- seed_honorific_005
- seed_japanese_leakage_011
- seed_ocr_noisy_007
- seed_oververbose_012
- seed_page_context_008
- seed_sfx_reaction_003
