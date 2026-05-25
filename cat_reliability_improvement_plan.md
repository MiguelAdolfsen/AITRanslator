# CAT Reliability Improvement Plan

## Goal
Raise CAT primary translation reliability to **above 90% approved CAT lines** on the 12-page diverse smoke set before treating CAT as a serious default-quality candidate.

For this goal, an approved CAT line means:
- CAT produced a non-empty English translation.
- CAT was not rejected by local validation.
- The output did not require Qwen fallback to become usable.
- The output is not obvious assistant chatter, prompt echo, untranslated Japanese, or repetitive boilerplate.

Current best CAT-only result:
- Run: `quality-runs/cat-second-retry-safe-salvage-norender-20260525`
- CAT-used lines: `88`
- CAT rejected: `6`
- CAT approved: `82 / 88`, about `93.2%`
- Ellipsis outputs: `6`
- Suspected bad translations: `7`
- Method: HF-style CAT GGUF primary, source-only retry, then strict incomplete-fragment retry only after retry failure, with conservative quoted-translation salvage.

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

Previous strict HF-chat baseline before safe salvage/second retry:
- Run: `quality-runs/cat-current-strict-baseline-norender-20260525`
- CAT-used lines: `88`
- CAT rejected: `11`
- CAT approved: `77 / 88`, about `87.5%`
- Ellipsis outputs: `11`
- Suspected bad translations: `12`

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

### Hypothesis 6: Exact Transformers Runtime Is Not Practical Here
The creator example and public Space use Hugging Face Transformers with:
- the model chat template,
- a system/user message pair,
- `do_sample=False`,
- `num_beams=1`,
- decoding only tokens generated after the prompt,
- tight `max_new_tokens` for short fragments.

The existing GGUF/Ollama path already uses deterministic sampling controls, but it may still differ from the official Transformers runtime in chat-template handling, stop behavior, tokenization, or generation defaults.

Decision:
- The unquantized/Hugging Face runtime is too large for the current GPU target.
- Do not pursue the full Transformers benchmark for now.
- Keep CAT reliability work focused on quantized GGUF/Ollama.
- The useful lesson still applies: quantized CAT experiments should mimic deterministic decoding as closely as Ollama allows, with tight fragment token caps, stop strings, and stricter short-fragment prompts.

Quantized follow-up:
- Test a separate GGUF/Ollama short-fragment model profile or prompt path rather than trying to run the full HF model.
- Keep the existing HF-template Ollama model as the stable baseline until a quantized-only experiment beats it.

### Tested Hypothesis 7: Quantized Short-Fragment Prompt Path
Hypothesis:
- Remaining CAT failures are concentrated around short manga fragments.
- A separate GGUF/Ollama prompt with a lower token cap might reduce assistant chatter without needing the full unquantized Transformers runtime.

Implementation tested:
- Opt-in short-fragment detection for lines with `1-10` Japanese characters.
- Fragment-specific prompt path.
- Fragment `num_predict` defaulted to `32`.
- Normal-length lines kept the existing CAT prompt.

Results:
- `quality-runs/cat-fragment-prompt-v2-norender-20260525`
  - Initial instruction-heavy fragment prompt appeared to reach `80 / 88`, but this was invalid because several accepted lines were prompt echoes or assistant chatter.
  - After tightening validation, the same run was effectively worse than baseline: `76 / 88`, with `12` CAT rejections.
- `quality-runs/cat-fragment-simple-norender-20260525`
  - Simpler creator-style fragment prompt dropped further to `74 / 88`, with `14` CAT rejections.

Decision:
- Revert the fragment prompt path.
- Keep the stricter generic validation for prompt echo and fragment-chatter patterns.
- Short fragments are still the main problem, but prompt-only specialization made GGUF CAT more chatty, not less.

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
   - Result: failed. Run `quality-runs/cat-context-retry-norender-20260525` stayed at `77 / 88`, about `87.5%` approved. It did not reduce final CAT rejections and added no useful rescue behavior.
   - Decision: revert. The extra context appears to make malformed/prompt-fragment failures more likely without improving the accepted count.

4. **Trimmed-source CAT retry**
   - For rejected CAT lines only, retry with leading/trailing ellipsis, dash, and punctuation noise stripped from the retry prompt.
   - Keep validation against the original OCR source.
   - Result: failed. Run `quality-runs/cat-trimmed-retry-norender-20260525` dropped to `75 / 88`, about `85.2%` approved, with `13` CAT rejections and `14` suspected bad translations.
   - Decision: revert. Punctuation trimming removed useful manga tone/context and increased untranslated-Japanese retry failures.

5. **Translation-focused CAT system prompt**
   - Keep the HF-style chat template but replace `SYSTEM "You are a helpful assistant."` with a stricter manga-translator system prompt.
   - Hypothesis: remaining chatter/refusal failures are caused by the model role behaving too much like a general assistant.
   - Result: failed. Run `quality-runs/cat-translator-system-norender-20260525` dropped to `63 / 88`, about `71.6%` approved, with `25` CAT rejections and `25` suspected bad translations.
   - Decision: revert. The creator-style helpful-assistant system is unexpectedly better for this GGUF/Ollama setup.

6. **CAT bypass + Qwen fallback routing**
   - This was tested briefly as a full-pipeline idea, but it is not the current CAT reliability goal.
   - Run `quality-runs/cat-bypass-q8-hybrid-norender-20260525` showed useful final-pipeline behavior, but it relies on Qwen and therefore does not count as CAT-alone reliability.
   - Decision: do not use this as proof that CAT is reliable. Keep the CAT-alone benchmark as the main target.

7. **Numbered mini-batch CAT**
   - Send the page's ordered lines together and parse numbered English output back onto CAT-rejected lines.
   - Hypothesis: nearby page text would give CAT enough context to avoid assistant chatter on short fragments.
   - Result: failed as a reliability solution. Run `quality-runs/cat-numbered-batch-norender-20260525` appeared to reach `88 / 88` by local validation, but manual inspection showed unacceptable line mixing and hallucinated rescues:
     - `きめらちょうかんだ` -> `It's a beautiful day today.`
     - `わたしがひみつそしき〈ぴーつー〉のぼす` -> `I'm going to climb the secret "P-Twin" mountain.`
     - noisy credit text was turned into plausible-looking names.
   - Decision: revert the batch retry code. The experiment proves page batching can suppress obvious chatter, but it replaces it with harder-to-detect hallucinations. It should not count toward CAT-alone reliability.

8. **Conservative CAT output salvage**
   - Hypothesis: a few CAT chatter responses contain a valid translation embedded after a clear marker, especially `English:`.
   - First broad salvage result:
     - Run: `quality-runs/cat-salvage-norender-20260525`
     - CAT approved appeared to improve to `82 / 88`, but manual inspection found unsafe accepted outputs:
       - `ガスがいて` -> `I have gas.`
       - `きめらちょうかんだ` -> `the time for the decision.`
     - Decision: reject broad salvage as unsafe.
   - Marker-only salvage result:
     - Run: `quality-runs/cat-salvage-marker-only-norender-20260525`
     - CAT approved `78 / 88`, about `88.6%`
     - This safely rescued `確かに．．．！` -> `Indeed...!`.
   - Safer explanatory salvage result:
     - Run: `quality-runs/cat-safe-explanatory-salvage-norender-20260525`
     - CAT approved `79 / 88`, about `89.8%`
     - It also rescued `あっかれん！` from an explanatory retry by extracting a quoted candidate only when CAT did not mention typos, unclear text, or "if you mean..." uncertainty.
   - Decision: keep conservative salvage. It is generic and low-risk when paired with local validation.

9. **Second strict incomplete-fragment retry**
   - Hypothesis: after CAT primary and source-only retry fail, a final stricter prompt can make CAT produce a direct translation or a salvageable explanatory answer for incomplete manga fragments.
   - Implementation tested:
     - Only runs after primary and first retry fail.
     - Prompt:
       ```text
       Translate exactly. If the source is incomplete, translate the incomplete fragment. Never ask for clarification.
       Japanese: "{source}"
       English:
       ```
     - Accepted only if the normal CAT validation and safe salvage pass.
     - Remains CAT-only; no Qwen or other translator is used.
   - Result:
     - Run: `quality-runs/cat-second-retry-safe-salvage-norender-20260525`
     - CAT-used lines: `88`
     - CAT rejected: `6`
     - CAT approved: `82 / 88`, about `93.2%`
     - Ellipsis outputs: `6`
     - Suspected bad translations: `7`
   - Accepted second-retry rescues:
     - `胸の内を明かしてくれると` -> `If you could reveal your true feelings.`
     - `残念な` -> `unfortunate`
     - `目立ってはいけない．．．` -> `You should not stand out...`
   - Still rejected, correctly:
     - `ガスがいて`
     - noisy credit/name OCR
     - `きめらちょうかんだ`
     - `わたしがひみつそしき〈ぴーつー〉のぼす`
   - Decision: keep the second retry as normal CAT behavior, with `MANGA_CAT_SECOND_RETRY=0` available to disable it for comparison runs.

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

Keep:
- HF-style CAT GGUF/Ollama model.
- Source-only retry.
- Second strict incomplete-fragment retry.
- Conservative marker/explanatory salvage.
- Strict chatter/prompt/Japanese validation.

Do not use:
- raw CAT by default.
- numbered mini-batch CAT.
- broad quote salvage.
- Qwen fallback as evidence that CAT itself is reliable.

Next best implementation target:
**Better handling of the remaining rejected OCR-risk lines**, especially noisy credits, katakana/name fragments, and OCR-corrupted playful phrases. These should probably be routed or reviewed by source-type rules rather than forced through CAT.
