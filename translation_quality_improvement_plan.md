# Translation Quality And Coherence Plan

## Summary
The next quality work should focus on text-side review and targeted correction rather than broad vision use. Vision facts remains experimental because it can help context but can also over-bias translation. The safer path is to make bad translations easier to inspect, then add focused review and deterministic rejection rules for recurring failures.

## Priorities
- **Translation review report**
  - Generate HTML/CSV from `.ocr.json` files.
  - Show source OCR, final text, OPUS baseline, Qwen candidate, verifier choice/reason, repair output, fallback output, context before/after, visual facts, and warning flags.
  - Highlight suspicious lines so we can review a chapter quickly without opening every image/debug JSON manually.

- **Translator + critic verifier**
  - Keep Qwen3.5 Q4 as the primary translator.
  - Add an optional second Qwen model as a critic/verifier, not a rewriter.
  - Recommended model role: Qwen3 Q4 checks accepted translations for issue labels such as `awkward_literal`, `context_mismatch`, `omitted_term`, `invented_detail`, `name_drift`, `tone_mismatch`, and `untranslated_text`.
  - The critic should return compact structured JSON with issue labels, severity, and a short reason. It should not produce a replacement translation.
  - Run the critic only on suspicious or medium-risk lines by default, not every bubble.
  - Batch the workflow by model: translate with Qwen3.5 Q4, unload it, run critic checks with Qwen3 Q4, then unload it before any Q8 repair. Do not switch models per bubble.
  - If the critic flags a hard or medium issue, send the line to Q8 for a constrained repair with the original OCR, nearby context, baseline translation, current translation, and critic issue labels.
  - Accept Q8 repairs only when existing local validation and verifier checks pass.

- **Q8 coherence review**
  - Keep Q4 as primary translator.
  - Use Q8 only for rejected lines or accepted-but-suspicious lines, not every bubble.
  - Review problems like awkward phrasing, context mismatch, dropped names/terms, known bad patterns, and narration/dialogue confusion.
  - Do not let Q8 rewrite lines unless existing local validation accepts the result.

- **Context-aware verification**
  - Keep initial translation close to the target bubble.
  - Give richer context to the verifier/repair pass, where it is safer to ask whether a line fits surrounding dialogue.
  - Avoid full page or chapter translation as the default.

- **Small consistency memory**
  - Track recurring names, bracket/code terms, and honorific style within a run.
  - Use it as verifier context and sanity checks, not as unrestricted story memory.

- **Bad-pattern rules**
  - Keep only generic rejection/postprocess rules in the default pipeline:
    - malformed JSON, markup, thinking tags, non-English output, and Japanese fallback text
    - malformed hyphen chains, repeated loops, stage-direction-only output, and overlong short-line translations
    - unnecessary gendering for generic group address
    - dropped code-like bracket terms such as `〈PII2〉`, while allowing noisy ruby/bracket text
  - Move series-specific name fixes, plot vocabulary, and exact phrase rewrites into optional user glossaries instead of hidden defaults.

## Current Decision
The review report, initial broad bad-pattern rules, optional critic pass, and critic-guided Q8 repair path are in place. The critic remains experimental/off by default, but when enabled it now runs in batches and can constrain Q8 repair with source text, nearby Japanese context, nearby accepted English context, baseline/current translations, and critic issue labels. The critic now reviews nearly all accepted dialogue/narration lines while skipping phrasebook/SFX, unusable outputs, and likely credit/name-only lines; Q8 repair is still gated by hard issue labels only. Critic freeform reasons are advisory only, and Q8 repair requires local source/text evidence for risky labels such as omitted terms, name drift, invented details, and untranslated text. Context mismatch alone is report-only until we have stronger local evidence.

## Evidence-Guided Critic/Q8 Pass

Implemented: critic and Q8 repair now receive deterministic evidence packets instead of relying only on broad prompt context. The evidence layer extracts source features, translation preservation failures, and run-local consistency memory. Debug reports, quality summaries, and review reports expose evidence risk flags, evidence repair reasons, critic evidence, and Q8 evidence rejection reasons. Q8 can be triggered by evidence-backed failures and is rejected if it worsens source-feature preservation, consistency, or basic English quality.

The default guardrails were audited after the first evidence benchmark. Hidden sample-specific rules were removed from production code, including the Japanese-to-English semantic term dictionary, exact character-name spelling fixes, exact hallucination phrase rejects, exact target rewrites from the early test chapter, and genre-specific source cleanup vocabulary. The default evidence path now focuses on generic checks: numbers, code-like bracket terms, OCR-risk flags, consistency memory, broken English, untranslated Japanese, schema/JSON failures, and layout pressure.

Next fitting steps:

1. Re-run the 12-page diverse smoke benchmark after the de-bias audit.
2. Compare generic evidence-guided Q8 repairs against `quality-runs/diverse-smoke-20260524`.
3. Tighten evidence rules only if failures recur across chapters.
4. Keep series-specific fixes in optional glossaries, not default code.
5. Keep vision facts experimental until text-only evidence-guided repair is stable.

## Open Questions
- Whether Qwen3 Q4 is reliable enough as a critic compared with Qwen3.5 Q4.
- Whether the critic should check only accepted suspicious lines or also lines that already used repair.
- Which issue labels should trigger Q8 repair versus report-only review.
