# Vision Implementation Spec for a Manga Translator using Qwen3.5 Vision

**Audience:** Codex / implementation agent  
**Purpose:** Implement a vision-aware manga translation mode inspired by *Context-Informed Machine Translation of Manga using Multimodal Large Language Models* without relying on that project’s source code.  
**Runtime target:** Qwen3.5 visual understanding models, either through Alibaba Cloud Model Studio / DashScope OpenAI-compatible API or a local OpenAI-compatible server such as vLLM/SGLang/Transformers.  
**Primary recommendation:** implement **page-by-page multimodal translation using a numbered, text-removed page image** (`PBP-VIS-NUM`) as the default vision path, with a plain page-image variant (`PBP-VIS`) as a fallback.

---

## 1. Source basis and constraints

This document is based on the COLING 2025 paper by Lippmann et al., its public arXiv/ACL text and appendices, dataset descriptions, README-level project metadata, and official Qwen / Alibaba Cloud Model Studio documentation for Qwen3.5 visual understanding, structured JSON output, image limits, and Qwen3.5 local serving. It intentionally does **not** depend on the project’s source code.

Key source findings to preserve from the manga translation paper:

1. Manga translation is not just OCR + sentence translation. It requires visual context for ambiguous references, speaker identity, setting, social context, tone, sound effects, and split speech across bubbles.
2. The paper assumes that page text has already been detected/OCR’d; the multimodal LLM is given the recognized Japanese lines plus the manga page image as visual context.
3. Current multimodal LLMs should **not** be asked to perform the entire job from the raw page image alone. They may misunderstand non-Latin text or rely on unreliable OCR. The system should pass OCR/transcribed lines explicitly.
4. Page-level translation is better than isolated line translation. The model should translate all lines from one page in reading order in a single request.
5. Visual context improves automatic scores, with `PBP-VIS` and `PBP-VIS-NUM` being the strongest methods in the paper.
6. More raw long context is not automatically better. Full-volume or very long context performed worse than single-page visual approaches. Use compact summaries or glossaries only when needed.
7. Asking the model to explain how the image affects each translation helped ensure visual context was actually used. In production, store this explanation as internal metadata, not as user-facing output.

Qwen3.5-specific constraints to preserve:

1. Use a **vision-capable Qwen3.5 model**. Hosted defaults should be `qwen3.5-plus` for quality or `qwen3.5-flash` for latency/cost. Local deployments should use a Qwen3.5 checkpoint with a vision encoder and must not start the server in text-only / language-model-only mode.
2. Qwen3.5 can accept image input through OpenAI-compatible chat-completion content arrays using `image_url` objects. The adapter must also support base64 data URLs for small local images and public/object-storage URLs for larger images.
3. Qwen does not use OpenAI’s `detail: high` image setting. For Alibaba Cloud Model Studio, map high-detail manga needs to `vl_high_resolution_images=true` or tuned `max_pixels`.
4. For production JSON, use non-thinking mode. Qwen thinking mode may produce hidden or explicit reasoning text before the final answer, and Alibaba’s structured-output documentation says thinking mode does not support structured output.
5. Qwen Model Studio structured output is best treated as **JSON object mode plus local validation**, not as a guarantee that the returned object semantically satisfies the schema. Always validate with `jsonschema`, Ajv, Pydantic, or the project’s existing validator.
6. Do not expose or store Qwen chain-of-thought content. Store only the short `visual_evidence` field requested by this spec.

---

## 2. Design goals

The vision layer must help translation, not replace the OCR/text pipeline.

### Goals

- Use visual context to resolve translation ambiguities that text-only MT misses.
- Preserve one output translation per detected manga text line/region unless the existing application explicitly supports line merging.
- Maintain stable mapping from source region → OCR text → visual label → translation → typesetting region.
- Reduce mistranslations caused by poor speaker/context assumptions.
- Improve consistency of names, pronouns, honorifics, register, tone, and scene-specific references.
- Be robust when a page has many bubbles, unusual panel layouts, sound effects, or free-floating text.
- Produce structured, parseable output suitable for automatic insertion into the translator pipeline.
- Keep cost and latency controlled by using one page per multimodal request by default.

### Non-goals

- Do not implement end-to-end translation from image only.
- Do not rely on the multimodal model to OCR Japanese text.
- Do not send whole volumes to the model by default.
- Do not expose model “reasoning” to end users. Store concise visual evidence/notes for debugging and QA.
- Do not assume visual context always improves every line. It should be treated as evidence, not as authority over the source text.

---

## 3. Recommended vision mode

Implement these modes, with `numbered_page` as the default.

| Mode | Paper analogue | Input to model | Use when | Default? |
|---|---|---|---|---|
| `text_page` | `PBP` | Ordered OCR lines only | Baseline/fallback, no vision model available | No |
| `page_image` | `PBP-VIS` | Ordered OCR lines + original page image | Numbered image generation fails or masking damages the page | Fallback |
| `numbered_page` | `PBP-VIS-NUM` | Ordered OCR lines + image where source text is removed/reduced and text regions are numbered | Normal production vision mode | **Yes** |
| `rolling_summary_page` | `VBP-VIS-COD`-like | Current page image + ordered OCR lines + short story summary/glossary | Long-running consistency problems across pages | Optional |
| `multi_page_window` | `VBP-VIS-3P`-like | Current page plus previous/next page images and lines | Rare diagnostics only | No |
| `full_volume` | `VBV-VIS` / `VBP-VIS-ALL`-like | Whole volume | Avoid except experimentation | No |

### Why `numbered_page` should be default

`PBP-VIS-NUM` removes or suppresses the original text in the page image and replaces each detected text region with a number corresponding to the OCR line list. This makes it easier for the model to locate each line visually while discouraging it from trying to OCR the Japanese text in the image.

The model gets the information it needs from two channels:

- **Text channel:** exact OCR/transcription for each line, in correct reading order.
- **Vision channel:** page layout, panels, characters, facial expressions, speaker positions, objects, setting, visual action, and bubble location.

---

## 4. Pipeline architecture

Implement the vision translation pipeline as a page-level module.

```text
Input page image
  + detected text regions
  + OCR text per region
  + reading order
  + optional previous glossary/summary
        ↓
Vision preprocessor
  - normalize page image
  - create region map
  - create numbered page image
        ↓
Prompt/request builder
  - build ordered line list
  - attach image
  - request structured JSON
        ↓
Multimodal model call
        ↓
Response parser and validator
  - validate line count and IDs
  - validate language and no missing translations
  - repair/retry when needed
        ↓
Translation result store
  - translation per line_id
  - visual notes per line_id
  - speaker/situation metadata
        ↓
Existing cleaning/typesetting pipeline
```

---

## 5. Required data model

Use existing application models if available, but ensure these fields exist or can be derived.

### `PageInput`

```json
{
  "page_id": "string",
  "image_path": "string",
  "width": 0,
  "height": 0,
  "source_language": "ja",
  "target_language": "en",
  "lines": [
    {
      "line_id": "stable unique id",
      "order_index": 1,
      "source_text": "Japanese OCR/transcription",
      "bbox": { "x": 0, "y": 0, "w": 0, "h": 0 },
      "region_type": "speech_bubble | thought_bubble | narration | free_text | sfx | unknown",
      "panel_id": "optional panel id",
      "speaker_id_hint": "optional known character id",
      "ocr_confidence": 0.0
    }
  ],
  "context": {
    "story_summary_target_language": "optional compact summary",
    "glossary": [
      { "source": "string", "target": "string", "note": "string" }
    ],
    "speaker_profiles": [
      { "speaker_id": "string", "name": "string", "voice_notes": "string" }
    ]
  }
}
```

### `VisionPageArtifact`

```json
{
  "page_id": "string",
  "mode": "numbered_page",
  "image_path": "path/to/generated_numbered_page.png",
  "number_map": [
    {
      "number": 1,
      "line_id": "stable unique id",
      "order_index": 1,
      "bbox": { "x": 0, "y": 0, "w": 0, "h": 0 }
    }
  ],
  "preprocessing_warnings": ["string"]
}
```

### `PageTranslationResult`

```json
{
  "page_id": "string",
  "model": "string",
  "mode": "numbered_page",
  "target_language": "en",
  "lines": [
    {
      "line_id": "stable unique id",
      "number": 1,
      "source_text": "Japanese OCR/transcription",
      "translation": "target-language translation",
      "speaker": "short visual/situational speaker description or null",
      "situation": "short setting/action description or null",
      "visual_evidence": "short note explaining what visual context affected the translation",
      "confidence": "high | medium | low",
      "warnings": []
    }
  ],
  "page_summary_target_language": "compact page event summary for optional rolling context",
  "terminology_updates": [
    { "source": "string", "target": "string", "reason": "string" }
  ]
}
```

---

## 6. Vision preprocessing requirements

### 6.1 Create a numbered page image

For `numbered_page` mode:

1. Load the original page image.
2. For each detected text region in reading order:
   - Use the region’s bounding box or text mask.
   - Remove or suppress Japanese text enough that the model is not tempted to OCR it.
   - Place a clear number corresponding to the ordered line list.
3. Preserve as much visual context as possible:
   - faces
   - bodies and gestures
   - objects being referenced
   - panel boundaries
   - setting/background
   - bubble shape and location
   - action lines and visual effects where possible
4. Save the generated image as PNG or JPEG.
5. Store a `number_map` so output can be mapped back to original `line_id`s.

### 6.2 Masking strategy

Use the least destructive masking that prevents OCR leakage.

Preferred order:

1. If a clean text mask is available, inpaint/fill only the text glyph pixels.
2. If only a bounding box is available inside a speech bubble, fill the interior text area with white or the dominant bubble background.
3. For free-floating text or SFX over artwork, be careful: heavy masking can remove meaningful art. Prefer a semi-local mask around glyphs, or fall back to `page_image` mode if masking destroys the scene.
4. Draw the number label near the center of the original text region or in an adjacent safe area.
5. Ensure labels are legible at the model’s input resolution.

### 6.3 Number rendering rules

- Use the page reading order: `1..N`.
- Make labels high contrast and large enough for vision models.
- Avoid covering faces, hands, important objects, or visual effects.
- For very small regions, place the number just outside the region with a pointer/outline if possible.
- Keep a deterministic layout so repeated runs produce the same artifact.
- If two text regions overlap, offset labels and include that warning in `preprocessing_warnings`.

### 6.4 Image normalization

- Preserve the full page aspect ratio.
- Do not crop to individual bubbles by default. Full-page context is the main source of visual value.
- Remove irrelevant scanner borders if they waste resolution, but do not remove panels or margins that clarify layout.
- Use high-quality resizing. Avoid blur.
- Keep final image within provider file limits.
- For dense manga pages, prefer higher image detail settings when the API/model supports them.

---

## 7. Prompting requirements

Use one request per page. The prompt should explicitly say that the OCR/transcribed line list is authoritative for source text, and the image is for context.

### 7.1 Default `numbered_page` prompt template

Use this as the base prompt. It is intentionally stricter and more structured than the paper prompt.

```text
You are a professional manga translator.

Task: Translate the Japanese manga lines from this page into {target_language}.

Important:
- The source text is provided below as OCR/transcription. Treat it as authoritative.
- The attached page image is visual context only.
- The image has numbered text regions. Each number corresponds to one line in the list.
- Do not perform OCR from the image. Use the provided source_text values.
- Use the image to infer speaker, emotion, scene, action, objects, social context, and whether a term is a name/proper noun.
- Return exactly one translation for every input line_id.
- Return a single JSON object only. Do not include markdown fences, commentary, or `<think>` content.
- Keep the translation natural for manga dialogue.
- Preserve intended tone, register, humor, urgency, and emotion.
- Do not add explanations to the translation itself.
- If the image does not help a line, set visual_evidence to "not visually informed".

Context summary, if any:
{story_summary_target_language}

Glossary / prior decisions, if any:
{glossary_json}

Lines in reading order:
{lines_json}

Return JSON matching the requested schema.
```

### 7.2 `page_image` fallback prompt difference

If using the original unmasked page image, add:

```text
The original Japanese text may be visible in the image, but it may be hard to read or wrong if inferred visually. Do not use image OCR as source text. Use only the provided source_text fields as the source of what is said.
```

### 7.3 Prompt content for `lines_json`

Use compact but explicit line objects.

```json
[
  {
    "line_id": "p012_l001",
    "number": 1,
    "source_text": "もう帰るの？",
    "region_type": "speech_bubble",
    "ocr_confidence": 0.98
  },
  {
    "line_id": "p012_l002",
    "number": 2,
    "source_text": "うん",
    "region_type": "speech_bubble",
    "ocr_confidence": 0.95
  }
]
```

### 7.4 Required output schema

Use the schema below for local validation in every provider. With Qwen Model Studio, request JSON object mode and include this schema in the prompt; still validate locally because JSON object mode constrains the response format but does not replace semantic schema validation. With local vLLM/SGLang, use JSON-schema guided decoding if the serving stack supports it.

```json
{
  "type": "object",
  "additionalProperties": false,
  "required": ["page_id", "target_language", "lines", "page_summary_target_language", "terminology_updates"],
  "properties": {
    "page_id": { "type": "string" },
    "target_language": { "type": "string" },
    "lines": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": [
          "line_id",
          "number",
          "source_text",
          "translation",
          "speaker",
          "situation",
          "visual_evidence",
          "confidence",
          "warnings"
        ],
        "properties": {
          "line_id": { "type": "string" },
          "number": { "type": "integer" },
          "source_text": { "type": "string" },
          "translation": { "type": "string" },
          "speaker": { "type": ["string", "null"] },
          "situation": { "type": ["string", "null"] },
          "visual_evidence": { "type": "string" },
          "confidence": { "type": "string", "enum": ["high", "medium", "low"] },
          "warnings": { "type": "array", "items": { "type": "string" } }
        }
      }
    },
    "page_summary_target_language": { "type": "string" },
    "terminology_updates": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "required": ["source", "target", "reason"],
        "properties": {
          "source": { "type": "string" },
          "target": { "type": "string" },
          "reason": { "type": "string" }
        }
      }
    }
  }
}
```

### 7.5 Why include `visual_evidence`

The paper found that asking the model to explain how the image influenced the translation helped ensure that visual context was used. In this implementation:

- Keep `visual_evidence` short.
- Store it for QA/debugging.
- Do not insert it into the typeset page.
- Use it to detect hallucinated visual use. Example bad evidence: “the character is angry” when no character is visible.

---

## 8. Qwen3.5 model/API adapter requirements

Implement a provider-agnostic `VisionLLMClient` interface so the translator is not hardcoded to a single Qwen serving path.

### 8.1 Interface

```text
VisionLLMClient.translate_page(
  prompt: string,
  image_path: string,
  schema: JsonSchema,
  model_config: ModelConfig
) -> PageTranslationResult
```

### 8.2 Supported Qwen providers

Implement at least one of these adapters, but keep the interface provider-neutral.

| Adapter | Typical model names | Use when | Notes |
|---|---|---|---|
| `qwen_modelstudio_openai` | `qwen3.5-plus`, `qwen3.5-flash` | Hosted Alibaba Cloud Model Studio / DashScope OpenAI-compatible API | Best default if you already use DashScope. `qwen3.5-plus` is quality-first; `qwen3.5-flash` is faster/cheaper. |
| `qwen_modelstudio_dashscope` | `qwen3.5-plus`, `qwen3.5-flash` | You prefer the native DashScope SDK and local file paths | Useful for local image paths. Less portable than OpenAI-compatible calls. |
| `qwen_local_openai` | `Qwen/Qwen3.5-9B`, other Qwen3.5 vision-capable checkpoints | Local vLLM/SGLang/Transformers server | Must be launched with multimodal support. Do not use `--language-model-only` or equivalent for vision translation. |

### 8.3 Qwen model selection

Default to:

```yaml
vision:
  model_provider: qwen_modelstudio_openai
  model_name: qwen3.5-plus
```

Use `qwen3.5-flash` when cost/latency matters more than maximum translation quality. For local deployments, make `model_name` fully configurable because users may choose different Qwen3.5 sizes, quantizations, or serving frameworks.

Do not assume that a text-only Qwen model is sufficient. The selected model must accept image inputs and answer from image + text together.

### 8.4 OpenAI-compatible request shape for Qwen

For Model Studio OpenAI-compatible API or a local OpenAI-compatible server, send one user message whose content contains the image plus the prompt text.

```python
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY", "EMPTY"),
    base_url=os.environ.get("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": numbered_page_image_url_or_data_url
                },
            },
            {
                "type": "text",
                "text": prompt_text_including_the_word_JSON,
            },
        ],
    }
]

completion = client.chat.completions.create(
    model=model_name,
    messages=messages,
    temperature=0.2,
    top_p=0.8,
    extra_body={
        # Hosted Model Studio:
        "enable_thinking": False,
        "vl_high_resolution_images": True,

        # Local vLLM/SGLang uses a different switch for non-thinking mode:
        # "chat_template_kwargs": {"enable_thinking": False},
    },
    response_format={"type": "json_object"},
)
```

Adapter rules:

- The prompt must contain the word `JSON` when using Qwen JSON object mode.
- Parse `completion.choices[0].message.content` as JSON.
- Strip markdown fences only as a defensive repair step; the normal path should not produce them.
- Do not include chain-of-thought in conversation history. If a prior Qwen response contains `<think>...</think>`, remove it before saving or reusing the response.
- For Alibaba Cloud Model Studio, region-specific API keys and endpoints are required. Keep `base_url` configurable.

### 8.5 Image transport and limits

Support these transports in this order:

1. **Base64 data URL** for small generated numbered-page images. Use `data:image/png;base64,...` or `data:image/jpeg;base64,...` matching the real file type.
2. **Public/object-storage URL** for larger images, especially when using the OpenAI-compatible or HTTP APIs.
3. **Local file path** only when using the native DashScope SDK, because OpenAI-compatible and DashScope HTTP calls do not accept local paths directly.

Implementation rules:

- Keep the generated numbered image under provider limits before sending.
- For Model Studio, base64-encoded images must be under the documented encoded-size limit. Because base64 increases size, prefer public/object-storage URLs when the original generated image is near 7 MB or larger.
- Avoid sending pages above 8K resolution. Pre-scale client-side to a reasonable manga-page size rather than relying on server scaling.
- Preserve full-page aspect ratio and labels. Do not crop bubbles unless running a diagnostic fallback.
- Prefer PNG for clean numbered artifacts when file size allows; use JPEG at high quality when PNG is too large.

### 8.6 Qwen image-detail controls

Do **not** implement OpenAI-style `detail: "high"` as a Qwen parameter. Instead, keep a provider-neutral config field and map it per adapter.

```yaml
vision:
  image_detail: high
  qwen:
    vl_high_resolution_images: true
    max_pixels: null
```

Mapping:

| Neutral setting | Qwen Model Studio behavior |
|---|---|
| `image_detail: low` | `vl_high_resolution_images=false`, default or reduced `max_pixels` |
| `image_detail: auto` | leave provider defaults unless labels are too small |
| `image_detail: high` | `vl_high_resolution_images=true`, or increase `max_pixels` if not using high-resolution mode |

For dense manga pages, start with high-resolution mode enabled. If latency/cost becomes unacceptable, benchmark `max_pixels` values while checking that numbered region labels remain readable.

### 8.7 Structured output strategy for Qwen

Use this hierarchy:

1. **Hosted Model Studio:** `response_format={"type":"json_object"}` + prompt containing `JSON` + local schema validation.
2. **Local vLLM:** if supported by the installed vLLM version, use JSON-schema guided decoding / structured outputs through `extra_body`, then still validate locally.
3. **Fallback:** prompt-only JSON + local validation + repair prompt.

For Model Studio JSON object mode:

- Disable thinking mode.
- Do not set `max_tokens` unless the API version and model require it; Alibaba’s structured-output guidance warns that `max_tokens` can truncate JSON and cause parsing failures.
- Keep the schema in the prompt even when using `response_format`, because Qwen’s JSON object mode mainly enforces that the output is a JSON string/object.

For local vLLM, the exact parameter name may vary by version. Support both modern and older forms if your codebase already abstracts generation parameters:

```python
# Modern vLLM-style structured output, when available:
extra_body={
    "chat_template_kwargs": {"enable_thinking": False},
    "structured_outputs": {"json": page_translation_schema},
}

# Older vLLM-style guided JSON, when available:
extra_body={
    "chat_template_kwargs": {"enable_thinking": False},
    "guided_json": page_translation_schema,
}
```

### 8.8 Thinking mode policy

For this translator, production mode should be **non-thinking**:

```yaml
vision:
  qwen:
    enable_thinking: false
```

Reasons:

- The pipeline needs parseable JSON, not open-ended reasoning text.
- The requested `visual_evidence` field already captures the useful human-readable evidence without exposing chain-of-thought.
- Structured-output support is documented for non-thinking mode, while thinking mode can require streaming and a second JSON repair pass.

If a future experiment uses thinking mode for translation quality, isolate it behind an explicit experimental flag:

1. Call Qwen with thinking enabled and streaming.
2. Extract only the final answer, never the `<think>` content.
3. Validate JSON.
4. If JSON is invalid, repair with a non-thinking Qwen model using JSON object mode.
5. Never pass thinking content into translation memory, logs, or user-facing output.

### 8.9 Recommended Qwen generation settings

Start with deterministic settings and tune on the regression set.

```json
{
  "temperature": 0.2,
  "top_p": 0.8,
  "top_k": 20,
  "presence_penalty": 0.0,
  "repetition_penalty": 1.0,
  "enable_thinking": false,
  "response_format": "json_object_or_guided_json",
  "timeout_seconds": 120,
  "retry_count": 2
}
```

Notes:

- Qwen’s public examples often use higher temperatures for general tasks. For translation insertion into a manga pipeline, lower temperature is preferable because line mapping, JSON validity, and consistency matter.
- If translations become too literal or flat, try `temperature: 0.3` or `0.4` before changing the pipeline.
- Keep sampling config separate from provider transport config.

---

## 9. Response validation and repair

After every model call, validate before accepting.

### 9.1 Hard validation

Reject and retry/repair when:

- JSON cannot be parsed.
- Required fields are missing.
- `page_id` mismatch.
- Line count does not equal input line count.
- Any input `line_id` is missing or duplicated.
- Any `number` does not match the `number_map`.
- A translation is empty.
- Output contains only source Japanese for a normal dialogue line.
- Output language appears wrong.
- Model adds extra lines not in the input.

### 9.2 Soft validation

Flag for QA but do not always reject:

- `confidence` is `low`.
- `visual_evidence` contradicts known region type or page metadata.
- A named entity differs from prior glossary.
- A line is much longer than the bubble can likely hold.
- A translation includes explanatory notes or parentheses not present in source style.
- OCR confidence was low and model warns about source uncertainty.

### 9.3 Repair prompt

If the response is structurally invalid, run a text-only repair pass using the original invalid JSON and the expected schema. Do not resend the image unless the content is semantically wrong.

```text
Repair the following model output so it exactly matches the required JSON schema.
Do not change the translations unless required to restore missing line_id/number/source_text mapping.
Return only valid JSON.

Expected line_ids and numbers:
{expected_line_map_json}

Invalid output:
{invalid_output}
```

If semantic validation fails, retry the original multimodal request once with a stricter instruction naming the failure. For Qwen, run repair with non-thinking mode and JSON object mode enabled; do not use thinking mode for repair unless the repair result is treated as untrusted text and validated again.

---

## 10. Long-context strategy

Do **not** pass the whole volume by default. The paper found that longer raw context did not consistently improve translation and could hurt quality.

Use this hierarchy:

1. **Default:** current page only, with numbered page image.
2. **Add glossary:** names, recurring terms, honorific choices, catchphrases, character voice decisions.
3. **Add compact rolling summary:** short target-language summary of prior story events.
4. **Use previous/next page images only for diagnostics or special cases.**
5. **Avoid full-volume multimodal calls in production.**

### 10.1 Rolling summary fields

After each page, update:

```json
{
  "story_summary_target_language": "Concise summary of story so far, capped at N words.",
  "characters": [
    {
      "name": "string",
      "aliases": ["string"],
      "visual_description": "string",
      "voice_style": "string",
      "relationship_notes": "string"
    }
  ],
  "glossary": [
    {
      "source": "string",
      "target": "string",
      "type": "name | place | object | catchphrase | sfx | other",
      "reason": "string"
    }
  ]
}
```

### 10.2 Summary update prompt

Use a separate text-only summarization call, or update locally from the page result.

```text
Update the manga translation memory using the previous memory and this page translation result.
Keep it concise. Preserve only details that help future translation: names, relationships, speaker style, locations, unresolved plot facts, recurring terms, and catchphrases.
Do not include a full transcript.
Return JSON matching the memory schema.
```

---

## 11. What vision should specifically help with

Codex should implement QA hooks and tests around these vision benefits.

### 11.1 Speaker and pronoun inference

Japanese often omits explicit subjects and pronouns. Vision can identify who is speaking, their approximate age/gender presentation, facial expression, and relation to nearby characters. Use this to choose natural target-language pronouns and phrasing, but do not over-infer identity if unclear.

### 11.2 Proper noun vs common noun

The paper’s example highlights that visual/scene context can make a reading like a person’s name more plausible than a literal common-noun reading. Store named entities in the glossary after they appear.

### 11.3 Scene/object references

A line may refer to an object or event visible in the panel or a previous panel. Vision should help identify whether a complaint, joke, or deictic phrase refers to a TV, food, weapon, sign, door, weather, etc.

### 11.4 Tone and emotion

Facial expressions, body language, typography, and panel composition should guide tone: angry, embarrassed, deadpan, teasing, frightened, formal, casual, etc.

### 11.5 Sound effects and free text

For SFX/free text:

- Translate the meaning naturally.
- Preserve whether it is sound, motion, background text, handwritten note, sign, or narration.
- Include `region_type` and `visual_evidence` so the typesetting layer can decide styling.
- Avoid over-literal SFX if target-language manga convention favors a different rendering.

### 11.6 Split sentences across bubbles

Page-level translation lets the model see adjacent bubbles. It should translate each region separately but account for the full sentence or exchange.

---

## 12. Integration with existing translator

### 12.1 New modules/classes

Implement or adapt these components:

```text
vision/
  numbered_page_renderer        # creates numbered page artifacts
  vision_prompt_builder         # builds mode-specific prompts and schemas
  vision_llm_client             # provider-agnostic image-capable model client
  qwen_vision_llm_client        # Qwen3.5 / DashScope / local OpenAI-compatible adapter
  page_translation_validator    # validates and repairs model output
  translation_memory            # optional glossary/summary state
  vision_translation_service    # orchestrates the page-level workflow
```

### 12.2 Configuration

Add config similar to:

```yaml
vision:
  enabled: true
  mode: numbered_page
  fallback_mode: page_image
  model_provider: qwen_modelstudio_openai      # qwen_modelstudio_openai | qwen_modelstudio_dashscope | qwen_local_openai
  model_name: qwen3.5-plus                     # use qwen3.5-flash for cheaper/faster mode
  base_url: ${QWEN_BASE_URL:-https://dashscope-intl.aliyuncs.com/compatible-mode/v1}
  api_key_env: DASHSCOPE_API_KEY

  image_detail: high                           # adapter maps this to Qwen-specific controls
  image_transport: auto                        # auto | base64 | public_url | dashscope_local_path
  temperature: 0.2
  top_p: 0.8
  top_k: 20
  max_retries: 2

  use_structured_outputs: true
  structured_output_mode: json_object          # json_object for Model Studio; guided_json/structured_outputs for local vLLM if available
  store_visual_evidence: true
  update_translation_memory: true
  max_summary_words: 250
  mask_source_text: true
  render_number_labels: true
  fail_open_to_text_page: true

  qwen:
    enable_thinking: false
    vl_high_resolution_images: true
    max_pixels: null
    strip_think_content: true
    forbid_markdown_fences: true
```

### 12.3 Pipeline behavior

For each page:

1. Ensure OCR/text detection has produced ordered lines.
2. If no lines are detected, skip translation but keep page metadata.
3. Generate a numbered page artifact.
4. Build a page-level vision prompt.
5. Call the vision model.
6. Validate response.
7. If validation fails:
   - try repair;
   - retry multimodal request if needed;
   - fall back to `page_image`;
   - then fall back to `text_page` if vision remains unusable.
8. Save translations and visual metadata.
9. Update glossary/summary if enabled.
10. Pass translations to existing typesetting.

---

## 13. Testing plan

### 13.1 Unit tests

Create tests for:

- Numbered image generation preserves dimensions.
- Every input line gets exactly one number.
- `number_map` is deterministic.
- Labels do not render outside image bounds.
- Prompt includes every `line_id`, `number`, and `source_text`.
- Schema rejects missing/duplicated line IDs.
- Validator catches wrong line counts.
- Repair prompt preserves translations and restores mapping.
- Fallback from `numbered_page` to `page_image` to `text_page` works.

### 13.2 Integration tests with mocked VLM

Mock model responses:

- valid JSON with all lines;
- malformed JSON;
- missing line;
- duplicate line;
- wrong language;
- hallucinated extra line;
- low-confidence visual ambiguity.

Assert the service handles each case predictably.

### 13.3 Regression/evaluation set

Build a small local evaluation set of pages that need visual context. Include pages with:

- ambiguous pronouns/subjects;
- named entities that could be common nouns;
- objects visible in panel;
- emotional tone only visible through faces/body language;
- sound effects;
- split bubbles;
- dense multi-panel layouts;
- low OCR confidence.

Evaluate these modes against each other:

1. `text_page`
2. `page_image`
3. `numbered_page`
4. `numbered_page + glossary/summary`

### 13.4 Human QA categories

Use a manga-specific MQM-inspired checklist:

- Fluency: punctuation, spelling, grammar.
- Accuracy: omissions, additions, mistranslations, untranslated text.
- Proper nouns / terminology: wrong name, failed name detection, inconsistent term.
- Style: formality, awkwardness, boring/flat phrasing, wrong emotional tone.
- Other: anything else that harms reading.

Track severity: `minor`, `major`, `critical`.

---

## 14. Acceptance criteria

The implementation is acceptable when:

1. Vision mode can translate a page using image + ordered OCR lines.
2. The default vision mode creates and uses numbered page images.
3. The model response is strict JSON and maps exactly back to input `line_id`s.
4. The system never relies on model OCR as the source text.
5. The pipeline falls back cleanly when image generation or model calls fail.
6. Visual evidence is stored for QA but excluded from the user-facing translation.
7. Existing typesetting receives the same type of translation records it expects.
8. Regression pages with visual ambiguity show better or at least more contextually plausible translations than text-only mode.
9. Long-context inputs are compact summaries/glossaries, not raw full volumes.
10. All new components have unit tests and mocked integration tests.

---

## 15. Practical implementation notes

### 15.1 Use image context sparingly but deliberately

The paper’s best results came from single-page visual approaches. Do not assume adding more pages improves quality. The image should be the current page, not a large visual dump.

### 15.2 Keep OCR and vision separate

The page image tells the model where and why a line is said. The OCR line list tells it what is said. Mixing these responsibilities causes avoidable errors.

### 15.3 Store visual metadata

`visual_evidence`, `speaker`, and `situation` are valuable for debugging and future memory, even if not displayed.

### 15.4 Minimize prompt bloat

Use concise context. Large prompts increase cost and can degrade quality. Prefer:

- current page lines;
- one image;
- short glossary;
- short summary;
- strict output schema.

### 15.5 Make failures visible

Log the following per page:

```json
{
  "page_id": "string",
  "vision_mode": "numbered_page",
  "model": "string",
  "line_count": 0,
  "image_artifact_path": "string",
  "validation_errors": [],
  "fallback_used": false,
  "token_or_cost_estimate": "optional",
  "latency_ms": 0
}
```

---

## 16. Implementation checklist for Codex

Codex should implement in this order:

1. Add config flags for `vision.enabled`, `vision.mode`, model provider, model name, and fallback behavior.
2. Add/extend data models for page lines, bounding boxes, numbered artifacts, and page translation results.
3. Implement `numbered_page_renderer`:
   - input: page image + ordered line regions;
   - output: numbered image + number map.
4. Implement `vision_prompt_builder`:
   - supports `numbered_page`, `page_image`, and `text_page`;
   - emits strict JSON schema.
5. Implement `VisionLLMClient` abstraction.
6. Implement the Qwen3.5 adapter:
   - OpenAI-compatible request shape with `image_url` + prompt text;
   - Model Studio support for `qwen3.5-plus` / `qwen3.5-flash`;
   - local OpenAI-compatible support for vLLM/SGLang/Transformers;
   - base64 data URL and public URL image transport;
   - Qwen-specific high-resolution controls (`vl_high_resolution_images` / `max_pixels`);
   - non-thinking mode by default;
   - JSON object mode or guided JSON plus local schema validation.
7. Implement `page_translation_validator`.
8. Implement retry/repair/fallback logic.
9. Implement optional `translation_memory` update.
10. Integrate `vision_translation_service` into existing page translation flow.
11. Add unit tests and mocked integration tests.
12. Add a small regression evaluation set for visual ambiguity.
13. Document how to enable/disable vision mode.

---

## 17. References

1. Philip Lippmann, Konrad Skublicki, Joshua Tanner, Shonosuke Ishiwatari, Jie Yang. “Context-Informed Machine Translation of Manga using Multimodal Large Language Models.” COLING 2025, ACL Anthology. https://aclanthology.org/2025.coling-main.232/
2. arXiv version / HTML: https://arxiv.org/abs/2411.02589 and https://arxiv.org/html/2411.02589v2
3. Research Software Directory dataset summary: https://research-software-directory.org/software/4tu-source-code-and-data-underlying-the-publication-context-informed-machine-translation-of-manga-using-multimodal-large-language-models
4. Project README-level metadata and dataset description: https://github.com/plippmann/multimodal-manga-translation
5. Alibaba Cloud Model Studio, Image and Video Understanding / Qwen3.5 visual understanding: https://www.alibabacloud.com/help/en/model-studio/vision
6. Alibaba Cloud Model Studio, Structured Output for Qwen models: https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output
7. Alibaba Cloud Model Studio, Supported Models and Capabilities Overview: https://www.alibabacloud.com/help/en/model-studio/models
8. Qwen3.5 Hugging Face model card and usage guidance: https://huggingface.co/Qwen/Qwen3.5-9B
9. vLLM Qwen3.5 / Qwen3.6 usage recipe: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html
10. vLLM structured outputs documentation: https://docs.vllm.ai/en/latest/features/structured_outputs/
