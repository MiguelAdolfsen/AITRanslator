# Benchmark Schema

Each benchmark set is a folder containing:

```text
cases.jsonl
references.jsonl
README.md
```

Optional fake-mode fixture:

```text
fake_outputs.jsonl
```

## cases.jsonl

Required fields:

```text
schema_version
case_id
source_text
source_type
risk_labels
```

Optional fields:

```text
page_id
line_id
neighbor_source_before
neighbor_source_after
notes
```

## references.jsonl

Required fields:

```text
schema_version
case_id
expected_decision
forbidden_patterns
allowed_japanese_output
max_chars
max_words
```

`expected_decision` must be `accept` or `reject`.

Optional fields:

```text
acceptable_outputs
must_preserve_terms
notes
```

## fake_outputs.jsonl

Used only for harness tests or smoke runs:

```json
{"case_id": "short_ellipsis_001", "raw_output": "...No way."}
```

Current source types:

```text
ellipsis_fragment
short_fragment
punctuation_fragment
interrupted_speech
normal_dialogue
long_dialogue
name_or_term
honorific_name
sfx
metadata_or_noise
ocr_noise
credits_or_metadata
narration
```
