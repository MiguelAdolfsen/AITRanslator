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

## 15. Advanced Multi-Model Hallucination Controls

Implement these only after the baseline validator, retry policy, and prompt contract are working.

Preferred order:

```text
1. translator + verifier
2. visual-facts pass + translator
3. candidate ensemble + selector
4. quality-estimation gate
5. back-translation checker
6. local contrastive decoding, only if local model serving supports it
```

The second model should usually have **less freedom** than the first model.

Bad pattern:

```text
Model A translates.
Model B translates again.
Pick whichever sounds better.
```

Good pattern:

```text
Model A translates.
Model B checks grounding, additions, omissions, mapping, names, and glossary consistency.
Model B returns issue labels only.
The pipeline, not the model, decides whether to accept, retry, repair, or review.
```

---

## 16. Translator + Verifier Pattern

Use this as the default advanced architecture.

```text
Model A: Qwen3.5-VL translator
Input:
- numbered page image
- OCR/source lines
- glossary
- validated context

Output:
- translation JSON

Model B: verifier / critic
Input:
- source lines
- translation JSON
- glossary
- optional validated visual facts

Output:
- issue labels only
- no free rewriting
```

### Verifier allowed labels

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

### Verifier prompt contract

```text
You are not translating.
You are not improving style.
You are only checking whether each English translation is grounded in the Japanese source_text and allowed context.

Return JSON only.
Do not rewrite translations unless explicitly instructed by the caller.
Classify issues using only the allowed issue labels.
```

### Verifier output schema

```json
{
  "page_id": "string",
  "line_checks": [
    {
      "line_id": "string",
      "has_issue": false,
      "issue_labels": [],
      "severity": "none",
      "explanation": "string"
    }
  ],
  "page_decision": "accept"
}
```

Suggested enums:

```text
severity: none | soft_flag | hard_reject
page_decision: accept | accept_with_flags | retry | needs_review
```

### Implementation rules

- The verifier must not mutate translations.
- The verifier must not add new translations.
- The verifier must preserve line IDs.
- The verifier must use only source text, glossary, validated memory, and optional validated visual facts.
- If the verifier detects invented content, mark the line for retry or review.
- If verifier output is malformed, do not trust it; fall back to baseline validator behavior.

---

## 17. Visual-Facts Pass + Translator Pattern

Use this for risky manga pages where vision is likely to help but could also hallucinate.

```text
Step 1: Qwen3.5-VL extracts observable visual facts only.
Step 2: A translation model translates source_text using those facts.
Step 3: Validator/verifier checks the final output.
```

### Trigger conditions

Run the visual-facts pass when any of these are true:

```text
multiple nearby speakers
low OCR confidence
ambiguous bubble mapping
many small bubbles
speaker identity affects translation
pronoun choice depends on visual context
prior validator failure
prior verifier failure
page risk score above threshold
```

### Visual-facts prompt contract

```text
You are not translating.
You are not reading Japanese from the image.
You are only describing observable visual facts for each numbered line/bubble.

Allowed facts:
- bubble location
- nearby character
- visible emotion
- visible action
- bubble type
- whether the line appears shouted, whispered, narrated, or thought-like

Forbidden facts:
- invented names
- relationships
- motives
- backstory
- personality
- hidden thoughts
- OCR corrections
- dialogue not present in source_text
```

### Visual-facts schema

```json
{
  "page_id": "string",
  "visual_facts": [
    {
      "line_id": "string",
      "bubble_type": "unknown",
      "speaker_position": "unknown",
      "nearby_character_description": "unknown",
      "visible_emotion": "unknown",
      "observable_action": "unknown",
      "mapping_confidence": "low",
      "facts": []
    }
  ]
}
```

Suggested enums:

```text
bubble_type: unknown | speech | thought | narration | shout | whisper | sfx
mapping_confidence: low | medium | high
```

### Validation rules

Reject visual facts if:

```text
line_id is missing
line_id is unknown
extra line_id appears
facts contain invented names
facts contain relationships not already validated
facts contain source_text rewrites
facts contain translated dialogue
facts contain unsupported backstory or motive
```

### Translation pass usage

The translation model receives:

```text
source_text
validated visual_facts
glossary
validated memory
```

The translation model must still obey:

```text
source_text is authoritative
visual_facts are context only
visual_facts cannot introduce new meaning into the translation
```

---

## 18. Candidate Ensemble + Selector Pattern

Use only for risky pages or high-value batches because this increases cost.

Generate multiple candidates:

```text
Candidate A: text-only page-by-page translation
Candidate B: vision-assisted page-by-page translation
Candidate C: visual-facts-assisted translation
```

Then use a selector/verifier to choose or flag.

### Do not majority-vote blindly

Translation candidates often differ stylistically.

Use disagreement as a risk signal:

```python
if candidates_are_semantically_similar(candidates):
    accept_best_style_candidate()
else:
    verifier_result = run_verifier(candidates)
    if verifier_result.safe_candidate_exists:
        accept_selected_candidate()
    else:
        mark_needs_review()
```

### Selector scoring dimensions

```text
source faithfulness
no added meaning
no omitted meaning
glossary consistency
speaker consistency
visual compatibility
bubble compactness
natural English
```

### Selector output schema

```json
{
  "page_id": "string",
  "line_selections": [
    {
      "line_id": "string",
      "selected_candidate_id": "A",
      "decision": "accept",
      "issue_labels": [],
      "needs_review": false
    }
  ]
}
```

Suggested enums:

```text
decision: accept | accept_with_flags | needs_review
```

### Implementation rules

- The selector must not create a new translation unless explicitly running a constrained repair pass.
- If no candidate is clearly grounded, set `needs_review=true`.
- Candidate disagreement should raise risk score even when one candidate is selected.

---

## 19. Quality-Estimation Gate

Optionally add a reference-free MT quality-estimation model or scoring function.

Use it as a soft gate, not an automatic judge.

```text
Input:
- Japanese source_text
- English translation

Output:
- quality score
- optional risk classification
```

### Suggested behavior

```python
if qe_score < QE_REVIEW_THRESHOLD:
    add_risk_flag("low_confidence_translation")
    set_needs_review(True)

if qe_score < QE_RETRY_THRESHOLD:
    run_constrained_retry_or_repair()
```

### Rules

- Do not reject solely because of QE score.
- Use QE together with validator/verifier results.
- Tune thresholds on your own manga pages.
- Track false positives from short lines, SFX, slang, jokes, and intentionally loose localization.

---

## 20. Back-Translation Checker

Optionally use back-translation as a soft hallucination signal.

```text
Japanese source_text
-> English translation
-> Japanese back-translation
-> compare source_text with back-translation
```

### Use cases

Good for detecting:

```text
added details
omitted details
wrong named entities
wrong polarity
wrong speaker intent
```

### Rules

- Never use back-translation mismatch as automatic hard rejection.
- Treat mismatch as a review flag.
- Expect false positives for idioms, slang, short lines, jokes, and natural localization.

### Suggested flag behavior

```python
if backtranslation_similarity < BACKTRANSLATION_REVIEW_THRESHOLD:
    add_risk_flag("low_confidence_translation")
    set_needs_review(True)
```

---

## 21. Local Contrastive Decoding

Implement only if running local models with enough decoding control.

This is not expected to work through a normal hosted chat API.

### Source-contrastive decoding

Goal:

```text
Prefer tokens that depend on the real source_text.
Penalize tokens that remain likely when the source_text is corrupted.
```

Conceptual scoring:

```text
score(token) =
    log P(token | correct_source, image, context)
    - lambda * log P(token | corrupted_source, image, context)
```

Possible corrupted sources:

```text
shuffled source lines
wrong page source lines
blanked source text
random same-length Japanese text
```

### Visual contrastive decoding

Goal:

```text
Prefer visual facts that depend on the real image.
Penalize facts that remain likely under a degraded image.
```

Possible degraded images:

```text
blurred page image
masked text regions
blank page layout
wrong page image
```

### Recommended usage

Use contrastive decoding only for:

```text
visual-facts pass
high-risk pages
research/benchmark mode
local model deployments
```

Do not make this a required production dependency.

---

## 22. Adaptive Page Risk Scoring

Implement a page risk score to decide when advanced passes run.

### Input risk signals

```text
low OCR confidence
many bubbles
many small text regions
multiple nearby characters
ambiguous speaker mapping
unusual layout
validator soft flags
validator hard reject
verifier issue labels
candidate disagreement
low QE score
back-translation mismatch
```

### Example scoring

```python
def score_page_risk(page, validation, verifier=None, candidates=None) -> str:
    score = 0

    if page.min_ocr_confidence < 0.80:
        score += 2
    if page.bubble_count >= 8:
        score += 1
    if page.has_ambiguous_speaker_mapping:
        score += 2
    if validation.hard_reject:
        score += 3
    if validation.soft_flags:
        score += 1
    if verifier and verifier.has_hard_issue:
        score += 3
    if candidates and candidates.semantic_disagreement:
        score += 2

    if score >= 5:
        return "high"
    if score >= 2:
        return "medium"
    return "low"
```

### Routing policy

```python
if risk == "low":
    use_single_qwen_vl_translation()

elif risk == "medium":
    use_translator_plus_verifier()

elif risk == "high":
    use_visual_facts_pass()
    use_translator_plus_verifier()
    mark_remaining_uncertainty_for_review()
```

---

## 23. Recommended Advanced Pipeline

Implement this as the target architecture.

```python
def translate_page(page):
    base = qwen_vl_translate(
        source_lines=page.lines,
        numbered_image=page.numbered_image,
        glossary=page.glossary,
        generation_config=QWEN_TRANSLATION_GENERATION_CONFIG,
    )

    validation = validate_translation_output(page, base)

    if validation.hard_reject:
        base = retry_with_repair_prompt(
            page=page,
            previous_output=base,
            validation=validation,
            max_retries=MAX_TRANSLATION_RETRIES,
        )
        validation = validate_translation_output(page, base)

    risk = score_page_risk(page, validation)

    if risk == "low":
        return base

    verifier_result = verifier_check(
        source_lines=page.lines,
        translation=base,
        glossary=page.glossary,
        validated_memory=page.validated_memory,
    )

    if verifier_result.page_decision == "accept":
        return base

    if risk == "medium" and verifier_result.page_decision == "accept_with_flags":
        return attach_review_flags(base, verifier_result)

    visual_facts = qwen_vl_extract_visual_facts_only(
        source_lines=page.lines,
        numbered_image=page.numbered_image,
    )

    visual_facts = validate_visual_facts(page, visual_facts)

    candidate = qwen_text_translate_with_facts(
        source_lines=page.lines,
        visual_facts=visual_facts,
        glossary=page.glossary,
        validated_memory=page.validated_memory,
        generation_config=QWEN_TRANSLATION_GENERATION_CONFIG,
    )

    candidate_validation = validate_translation_output(page, candidate)

    selection = select_safe_translation_or_flag_review(
        page=page,
        candidates=[base, candidate],
        validations=[validation, candidate_validation],
        verifier_result=verifier_result,
        visual_facts=visual_facts,
    )

    return selection
```

### Pipeline invariants

```text
The validator always runs.
The verifier never freely rewrites.
The visual-facts pass never translates.
The translator never mutates source_text.
The selector never invents a new translation unless using constrained repair.
Uncertain facts never enter long-term memory.
```

---

## 24. Advanced Method Acceptance Criteria

Advanced hallucination controls are complete when:

```text
translator + verifier can run on medium-risk pages
visual-facts pass can run on high-risk pages
verifier returns issue labels only
visual-facts pass returns observations only
candidate disagreement can trigger review
QE/back-translation gates are soft signals only
page risk score controls routing
advanced passes never bypass baseline validation
memory promotion rules still apply
```

---

## 25. Minimal Tests

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

### Advanced multi-model methods

```text
verifier_returns_issue_labels_only
verifier_does_not_rewrite_translation
visual_facts_pass_rejects_translated_dialogue
visual_facts_pass_rejects_invented_relationship
candidate_disagreement_sets_review_flag
qe_low_score_sets_review_flag_not_hard_reject
backtranslation_mismatch_sets_review_flag_not_hard_reject
risk_score_routes_low_risk_to_single_translation
risk_score_routes_medium_risk_to_verifier
risk_score_routes_high_risk_to_visual_facts_pass
```

---

## 26. Acceptance Criteria

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
- advanced multi-model methods are gated by page risk score
- verifier cannot freely rewrite translations
- visual-facts pass cannot translate or invent facts
- candidate selection treats disagreement as a risk signal
- QE and back-translation checks are soft gates only

---

## 27. One-Line Implementation Rule

```text
When the model is uncertain, force it to become less specific, not more creative.
```
