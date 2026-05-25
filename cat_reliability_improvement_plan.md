# CAT Reliability Improvement Plan

## Goal
Raise CAT primary translation reliability to **above 90% approved CAT lines** on the 12-page diverse smoke set before treating CAT as a serious default-quality candidate.

For this goal, an approved CAT line means:
- CAT produced a non-empty English translation.
- CAT was not rejected by local validation.
- The output did not require Qwen fallback to become usable.
- The output is not obvious assistant chatter, prompt echo, untranslated Japanese, or repetitive boilerplate.

Current best CAT-only baseline:
- Run: `quality-runs/cat-only-hfchat-v6-20260525`
- CAT-used lines: `88`
- CAT rejected: `11`
- CAT approved: `77 / 88`, about `87.5%`
- Ellipsis outputs: `11`
- Suspected bad translations: `12`

Target:
- CAT rejected: `<= 8 / 88`
- CAT approved: `>= 80 / 88`
- Chatter/prompt failures should be near zero.

Previous baseline before the HF chat-template fix:
- Run: `quality-runs/cat-only-retry-20260525`
- CAT-used lines: `88`
- CAT rejected: `22`
- CAT approved: `66 / 88`, about `75%`
- Ellipsis outputs: `22`
- Suspected bad translations: `23`

## Tested Hypothesis 1: Raw Ollama Template

### Hypothesis
CAT looked like it was being harmed by chat-assistant wrapping. Many failures were assistant-like responses:
- "I don't see any Japanese text..."
- "Could you please provide..."
- "Translated by..."
- "Assist with the translation..."

The hypothesis was that CAT should be run as a raw completion model rather than through an Ollama model with a system prompt/chat-style template.

### Implementation Tested
- Created a separate raw CAT Ollama model:
  - `manga-cat-cat-translate-7b-q8-0-raw`
- Modelfile used:
  - `TEMPLATE """{{ .Prompt }}"""`
  - no `SYSTEM` prompt
- API generation used raw prompting.
- Prompt stayed short and completion-style.

### Result
The hypothesis failed.

Raw CAT run:
- Run: `quality-runs/cat-only-raw-20260525`
- CAT-used lines: `88`
- CAT rejected: `78`
- CAT approved: `10 / 88`, about `11%`
- Ellipsis outputs: `81`
- Suspected bad translations: `81`
- Main failure: `cat_verbose_or_repetitive`: `76`

Raw outputs often repeated phrases until the token cap, for example:
- repeated "Already!"
- repeated "Return to classroom?"
- repeated fragments from the source line

### Decision
Do **not** keep raw CAT as the default. The raw Ollama model was removed, and CAT was reverted to the previous non-raw Ollama path.

## What We Learned

The original CAT problem is not simply "Ollama chat wrapping is bad." The raw mode removed assistant chatter but made repetition much worse.

The current normal Ollama path is still better:
- It has assistant-chatter failures.
- But it produces many usable translations.
- Validation plus Qwen fallback can recover many failures.

The next likely improvement area is not raw prompting. It is **prompt shape, generation stopping, and source batching/short-line handling**.

## Tested Hypothesis 2: Use CAT-7B's Hugging Face Chat Template

### Hypothesis
The raw-template test failed, but the original Ollama model was still not matching the creator's instructions. The CAT-Translate-7B model card says the 7B model must use its chat template, and that the 7B template differs from the other CAT-Translate models.

The hypothesis was that CAT reliability would improve if the GGUF/Ollama path reproduced the 7B Hugging Face chat template instead of using a generic Ollama system prompt.

### Implementation Tested
Created a new Ollama model identity:
- `manga-cat-cat-translate-7b-q8-0-hfchat`

The generated Modelfile now uses:
```text
SYSTEM """You are a helpful assistant."""
TEMPLATE """<s><|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""
```

It also includes stop tokens:
```text
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
```

The prompt remains the creator-style user prompt:
```text
Translate the following Japanese text into English.

{source_text}
```

### Result
This hypothesis succeeded enough to keep.

First HF-chat run before stricter refusal validation:
- Run: `quality-runs/cat-only-hfchat-20260525`
- CAT-used lines: `88`
- CAT rejected: `7`
- CAT approved by local validation: `81 / 88`, about `92.0%`
- Ellipsis outputs: `7`
- Suspected bad translations: `8`

However, manual inspection found a few accepted lines that were still assistant-style refusal/clarification responses. Validation was tightened to catch:
- "I can't help with that"
- "could you please clarify"
- "provide more context"
- "I don't understand the Japanese/text"

Final stricter HF-chat run:
- Run: `quality-runs/cat-only-hfchat-v6-20260525`
- CAT-used lines: `88`
- CAT rejected: `11`
- CAT approved: `77 / 88`, about `87.5%`
- Ellipsis outputs: `11`
- Suspected bad translations: `12`

### Decision
Keep the HF-chat-template CAT Ollama model as the current CAT path. It is a clear improvement over the previous CAT baseline:
- rejected lines improved from `22` to `11`
- ellipsis outputs improved from `22` to `11`
- suspected bad translations improved from `23` to `12`

The strict 90% target is not reached yet, but this is the new baseline.

## Possible Tweaks To The Tested Solution

These are variations on the raw-template idea that might still be worth testing later, but they should not be assumed to work.

### 1. Raw Template With Strong Stop Sequences
Raw mode may have failed because it had no natural stop point.

Testable tweak:
- Keep raw/template-only model.
- Reduce `num_predict` sharply, for example `32` or `48`.
- Add stop strings if supported by the Ollama API/options:
  - `Japanese:`
  - `English:`
  - `Translation:`
  - newline after a completed sentence

Risk:
- May cut off valid longer translations.
- Might still repeat before hitting stop.

### 2. Raw Template With One-Shot Example
CAT may need a tiny demonstration to stabilize completion.

Example:
```text
Japanese:
何あれ

English:
What is that?

Japanese:
{source}

English:
```

Risk:
- Can bias outputs toward the example style.
- May leak the example into output.
- Needs validation for prompt echo.

### 3. Raw Template Only For Specific Failure Types
Raw mode performed badly overall, but it might still rescue a narrow subset.

Possible use:
- Only retry raw mode after normal CAT fails with `cat_chatter`.
- Never use raw mode for normal primary translation.

Risk:
- The tested raw retry was also poor, so this is low priority.

## More Promising Hypotheses

### Hypothesis 2: CAT Needs A Different Prompt For Short Manga Fragments
Many CAT failures are very short or fragmentary:
- `何あれ`
- `．．．そんな`
- `残念な`
- `―――では！！`
- names or credits

CAT may be less stable when the input is too short or lacks sentence context.

Possible solution:
- Keep normal CAT prompt for normal dialogue.
- Use special handling for short fragments:
  - phrasebook/SFX/name rules first,
  - or OPUS/Qwen fallback first,
  - or CAT with nearby context lines.

Test:
- Categorize CAT failures by source length and type.
- Run CAT only on lines above a minimum source length.
- Compare approval rate on eligible CAT lines.

### Hypothesis 3: CAT Needs Numbered Multi-Line Context
Instead of translating isolated short bubbles, CAT may behave better when given several ordered lines together.

Example:
```text
Translate each numbered Japanese line into concise English.

1. あっもう１２時だ
2. 一年生は先に休憩入って～
3. 教室戻る？

1.
```

Acceptance would still happen per line.

Risks:
- Structured output parsing failures.
- Line mixing.
- More prompt echo.

Mitigation:
- Only test on pages with short-line CAT failures.
- Require exact numbered output.
- Fall back to current single-line CAT on malformed batch output.

### Hypothesis 4: CAT Needs Lower Output Budget
Some bad CAT outputs are long assistant messages or repeated text. A lower token cap may convert catastrophic failures into short failures.

Test:
- Compare `num_predict=32`, `48`, `64`, `96`, and current `128`.
- Track:
  - CAT rejection rate,
  - truncation rate,
  - verbose/repetitive failures,
  - valid long translation failures.

This is a cheap experiment and should be prioritized.

### Hypothesis 5: CAT Should Not Translate Names/Credits/SFX
CAT struggles with names, credits, and non-dialogue fragments. These often do not need full neural translation.

Possible solution:
- Detect name-like katakana/person credits and pass through transliteration or phrasebook.
- Detect common SFX and use phrasebook.
- Detect credits/noisy metadata and skip or route to Qwen.

This is likely generic and not test-set overfitting if the rules are type-based, not phrase-specific.

## Next Experiments

Run these in order, keeping the benchmark fixed:

0. **Creator-prompt CAT retry**
   - Keep HF chat-template CAT primary.
   - Change CAT retry from `Japanese:\n...\nEnglish:` to the same creator/model-card prompt used by primary.
   - Hypothesis: some remaining retry failures are caused by the retry prompt not matching the 7B chat-template usage instructions.
   - Benchmark without rendering images.
   - Result: failed. Run `quality-runs/cat-retry-card-norender-20260525` dropped to `64 / 88`, about `72.7%` approved. Keep the previous `Japanese:\n...\nEnglish:` retry prompt.

1. **CAT token-budget sweep**
   - `num_predict`: `32`, `48`, `64`, `96`, `128`
   - Keep the current normal Ollama model, not raw.
   - Pick the setting with the best CAT approval rate without truncating valid lines.
   - Implementation note: `MANGA_CAT_NUM_PREDICT` now controls CAT GGUF output budget for benchmark experiments. Benchmarks skip image rendering by default.
   - Result so far:
     - `quality-runs/cat-token64-norender-20260525`: `77 / 88`, `87.5%`
     - `quality-runs/cat-token32-norender-20260525`: `77 / 88`, `87.5%`
   - Decision: no improvement. Keep default `128` unless a later test shows a reason to lower it.

2. **Short-line routing**
   - Identify CAT failures by source length/type.
   - Try bypassing CAT for obvious names/SFX/credits/ellipsis fragments.
   - Measure CAT approval rate on the lines CAT still handles.
   - Implementation note: `MANGA_CAT_BYPASS_RISKY_SOURCE=1` enables this experiment. It is off by default because CAT-only output otherwise gets more ellipses.
   - Result: `quality-runs/cat-bypass-risky-optin-norender-20260525`
     - CAT-used lines dropped from `88` to `70`
     - CAT approved `65 / 70`, about `92.9%`
     - CAT rejected `5`
     - But ellipsis outputs rose to `23`
   - Decision: useful as a CAT+Qwen routing strategy, not acceptable as unconditional CAT-only behavior. Keep behind env flag for targeted/hybrid experiments.

3. **Contextual CAT retry**
   - For normal CAT failures only, retry with neighboring source lines.
   - Keep primary CAT unchanged.
   - Accept only if validation passes.

4. **Numbered mini-batch CAT**
   - Only after the above.
   - Test on the same 12-page set and manually inspect line mixing.

## Tracking Requirements

Every CAT experiment should report:
- CAT-used lines
- CAT approved lines
- CAT approval percentage
- CAT rejected lines
- rejection reasons
- CAT retry attempted/accepted/rejected
- ellipsis outputs
- suspected bad translations
- examples of accepted CAT outputs that still read wrong

Benchmark folders should use stable names like:
- `quality-runs/cat-token32-20260525`
- `quality-runs/cat-token48-20260525`
- `quality-runs/cat-short-routing-20260525`
- `quality-runs/cat-context-retry-20260525`

## Current Recommendation

Keep the current normal CAT Ollama path, CAT retry scan, and strict validation.

Do not use raw CAT by default.

Next best implementation target:
**CAT token-budget sweep**, because it is local, simple, measurable, and directly attacks the verbose/repetitive failure mode without changing translation logic.
