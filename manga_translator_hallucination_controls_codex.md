# Codex Task: Hallucination Controls for Qwen3.5 Manga Translation

Implement hallucination-reduction controls for the manga translation pipeline.

This document is for Codex only. Treat every item below as implementation guidance, not explanatory prose.

---

## 1. Non-Negotiable Runtime Policy

The translator must follow these rules:

```text
The model translates only the provided source_text fields.
The page image is visual context only.
The image must never be treated as a source of new dialogue, narration, SFX, or OCR text.
```

Hard requirements:

- One output item per input line.
- Preserve every `line_id` exactly.
- Preserve every `source_text` exactly.
- Do not allow the model to add lines.
- Do not allow the model to silently correct OCR.
- Do not allow invented speaker names.
- Do not allow invented relationships, motives, locations, backstory, or gendered assumptions.
- When uncertain, prefer a neutral translation and mark the line for review.

---

## 2. Recommended Module Boundaries

Create or update these components:

```text
translator/
  vision_prompt.py
  schemas.py
  hallucination_validator.py
  retry_policy.py
  review_flags.py
  generation_config.py
  tests/
    test_hallucination_validator.py
    test_retry_policy.py
    test_prompt_contract.py
```

Use equivalent paths if the project already has a different structure.

---

## 3. Generation Config

Use deterministic defaults for production translation.

```python
QWEN_TRANSLATION_GENERATION_CONFIG = {
    "temperature": 0.1,
    "top_p": None,
    "top_k": 20,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
    "seed": 12345,
    "enable_thinking": False,
}
```

Rules:

- Disable thinking mode for production JSON output.
- Prefer low `temperature`.
- Do not tune both `temperature` and `top_p` aggressively at the same time.
- Keep generation reproducible where the endpoint supports `seed`.

---

## 4. Prompt Contract

Add this block to the production prompt.

```text
Anti-hallucination rules:
- Translate only the provided source_text fields.
- Do not add dialogue, narration, sound effects, thoughts, or names not present in the input line list.
- Do not invent a speaker name. Use "unknown" unless the speaker is visible, stated in the text, or present in validated glossary/context.
- Do not invent gender, age, relationship, location, motive, or backstory.
- If uncertain, keep the English wording neutral.
- Do not use the image to "correct" Japanese OCR.
- If OCR seems suspicious, preserve source_text and set needs_review=true with an issue note.
- visual_evidence must mention only observable facts from the image.
- If no observable visual fact helped, write "not visually informed".
- If a required decision cannot be grounded, choose the least specific natural translation and set translation_confidence="low".
- Return JSON only.
```

Also include this task instruction:

```text
Return exactly one translation object for each input line.
The order should match the input order.
Every line_id and source_text must be copied exactly.
```

---

## 5. Output Schema

Implement a schema equivalent to this.

```json
{
  "type": "object",
  "required": ["page_id", "translations"],
  "additionalProperties": false,
  "properties": {
    "page_id": { "type": "string" },
    "translations": {
      "type": "array",
      "items": {
        "type": "object",
        "required": [
          "line_id",
          "source_text",
          "translation",
          "speaker",
          "speaker_confidence",
          "visual_evidence_type",
          "visual_evidence",
          "translation_confidence",
          "needs_review",
          "risk_flags"
        ],
        "additionalProperties": false,
        "properties": {
          "line_id": { "type": "string" },
          "source_text": { "type": "string" },
          "translation": { "type": "string" },
          "speaker": { "type": "string" },
          "speaker_confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"]
          },
          "visual_evidence_type": {
            "type": "string",
            "enum": ["none", "weak", "direct"]
          },
          "visual_evidence": { "type": "string" },
          "translation_confidence": {
            "type": "string",
            "enum": ["low", "medium", "high"]
          },
          "needs_review": { "type": "boolean" },
          "risk_flags": {
            "type": "array",
            "items": {
              "type": "string",
              "enum": [
                "speaker_uncertain",
                "ocr_suspect",
                "visual_ambiguity",
                "possible_name_invention",
                "possible_gender_inference",
                "glossary_conflict",
                "mapping_uncertain",
                "bubble_fit_risk",
                "low_confidence_translation"
              ]
            }
          }
        }
      }
    }
  }
}
```

---

## 6. Qwen Structured Output Policy

Preferred order:

1. Use strict `json_schema` mode if supported by the active Qwen endpoint/model.
2. Otherwise use JSON-object mode and local schema validation.

Implementation requirements:

- The prompt must explicitly contain the word `JSON`.
- Validate every response locally.
- Reject invalid schema responses.
- Retry only with a constrained repair prompt.
- Do not pass unvalidated model output downstream.

---

## 7. Hallucination Validator

Implement `validate_translation_output(input_page, model_output) -> ValidationResult`.

The validator must classify errors as:

```python
class ValidationSeverity:
    HARD_REJECT = "hard_reject"
    SOFT_FLAG = "soft_flag"
```

### Hard Reject Rules

Reject output when any of these are true:

```text
invalid_json
schema_invalid
line_count_mismatch
missing_line_id
duplicate_line_id
unknown_line_id
line_order_mismatch
source_text_mismatch
empty_translation
invalid_enum_value
extra_translation_object
visual_evidence_claims_fact_when_type_none
```

Specific checks:

- `len(output.translations) == len(input.lines)`
- each input `line_id` appears exactly once
- no output `line_id` outside the input set
- output order matches input order unless the existing pipeline explicitly allows reordering
- `output.source_text == input.source_text`
- `translation.strip()` is non-empty
- `visual_evidence_type == "none"` requires `visual_evidence == "not visually informed"` or an equivalent non-claim phrase

### Soft Flag Rules

Flag but do not necessarily reject when any of these are true:

```text
speaker_is_specific_but_confidence_low
speaker_unknown_but_visual_evidence_direct
translation_confidence_low
needs_review_true
risk_flags_non_empty
ocr_confidence_low
glossary_conflict
translation_too_long_for_bubble
possible_name_invention
possible_gender_inference
```

---

## 8. Speaker and Identity Guardrails

Implement helper checks.

### Speaker default

```python
DEFAULT_SPEAKER = "unknown"
```

Rules:

- Use `unknown` unless identity is grounded.
- A named speaker is allowed only if supported by at least one:
  - source text
  - validated glossary
  - validated character memory
  - explicit user/editor-provided metadata
  - direct visual mapping plus validated context

### Do not persist uncertain identity

Never write these to long-term memory automatically:

- low-confidence speaker guesses
- inferred names
- inferred relationships
- inferred gender
- visual evidence strings
- critic suggestions

---

## 9. OCR Safety

Input OCR/transcription is authoritative.

Validator and prompt must enforce:

- model may not rewrite `source_text`
- model may not translate imagined text from the image
- suspicious OCR should be flagged, not corrected

When OCR is suspicious:

```python
line.needs_review = True
line.risk_flags.append("ocr_suspect")
```

Do not mutate source text inside the translation step.

---

## 10. Retry Policy

Implement bounded retries.

```python
MAX_TRANSLATION_RETRIES = 2
```

Retry only for hard rejects.

Use a repair prompt like:

```text
Your previous response was rejected.

Reasons:
{validation_error_list}

Return corrected JSON only.

Requirements:
- preserve every line_id exactly
- preserve every source_text exactly
- return exactly one item per input line
- do not add any new dialogue, narration, SFX, names, or facts
- do not use the image as OCR
```

After retry exhaustion:

- fall back to text-only translation, or
- return the safest available translation with `needs_review=true`, depending on existing pipeline behavior.

Do not enter an unbounded repair loop.

---

## 11. Optional Visual Facts Pass

For difficult pages, implement an optional pre-translation pass.

Trigger conditions:

```text
multiple nearby speakers
low OCR confidence
ambiguous bubble mapping
many small bubbles
speaker identity needed for natural translation
prior validation failures
```

Visual facts pass output should contain only observable facts:

```json
{
  "line_id": "string",
  "observable_facts": [
    "bubble number 3 is near the character on the left",
    "the character appears shocked",
    "the bubble appears to be shouted speech"
  ]
}
```

Rules:

- No translation in this pass.
- No invented names.
- No inferred relationships.
- No hidden motivations.
- Feed only validated visual facts into the translation pass.

---

## 12. Optional Critic Pass

Implement an optional critic pass for QA.

The critic must classify issues only.

Allowed issue labels:

```text
addition
omission
invented_name
invented_relationship
unnecessary_gendering
wrong_line_mapping
source_text_mismatch
glossary_conflict
untranslated_japanese
bubble_fit_risk
```

Rules:

- The critic should not freely rewrite translations.
- If the critic finds an issue, either:
  - flag for review, or
  - run a constrained repair pass.
- Do not store critic claims in long-term memory.

---

## 13. Memory Contamination Rules

Before writing anything to glossary, character memory, or translation memory:

```python
def can_promote_to_memory(fact) -> bool:
    return (
        fact.supported_by_source_text
        or fact.in_validated_glossary
        or fact.confirmed_by_editor
        or fact.confirmed_repeatedly_across_pages
    )
```

Never promote:

```text
low-confidence speaker guesses
inferred gender
inferred relationships
unvalidated names
visual_evidence strings
critic suggestions
uncertain terminology
```

---

## 14. Bubble Fit Priority Order

When compact output is needed, preserve this priority:

```text
1. faithfulness
2. no added meaning
3. natural English
4. bubble fit
5. style polish
```

Never shorten by adding interpretation.

---

## 15. Minimal Tests

Add tests for these cases.

### Schema and mapping

```text
rejects_invalid_json
rejects_missing_translation_item
rejects_extra_translation_item
rejects_duplicate_line_id
rejects_unknown_line_id
rejects_source_text_mutation
rejects_empty_translation
```

### Hallucination controls

```text
rejects_added_dialogue_line
flags_specific_speaker_with_low_confidence
flags_possible_name_invention
flags_possible_gender_inference
flags_low_translation_confidence
flags_ocr_suspect
does_not_store_uncertain_speaker_in_memory
```

### Visual evidence

```text
rejects_visual_evidence_fact_when_type_none
accepts_not_visually_informed_when_type_none
accepts_direct_visual_evidence_when_type_direct
```

### Retry policy

```text
retries_hard_reject_once_or_twice
does_not_retry_soft_flags_unless_configured
falls_back_after_retry_exhaustion
repair_prompt_contains_rejection_reasons
```

---

## 16. Acceptance Criteria

Implementation is complete when:

- production prompt contains the anti-hallucination contract
- Qwen generation config disables thinking mode
- structured output is used where available
- local schema validation runs on every response
- hard rejects trigger bounded retries
- soft flags are surfaced for review
- source text cannot be silently changed
- line mapping is exact
- uncertain identity is not persisted to memory
- tests cover validator, retry policy, prompt contract, and memory promotion rules

---

## 17. One-Line Implementation Rule

```text
When the model is uncertain, force it to become less specific, not more creative.
```
