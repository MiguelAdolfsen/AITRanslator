from __future__ import annotations

import logging
import re

from .config import PipelineConfig
from .detect_types import TextBlock
from .grouping import context_for_block
from .line_identity import (
    translation_for_page_order_item,
    translation_key_for_block,
)
from .logging_utils import shorten
from .page_types import PreparedPage
from .qwen_validation import source_has_bracket_term, translation_preserves_bracket_term
from .text_filter import count_japanese_chars, suspected_bad_translation, unusable_translation_reason
from .translated_state import TranslationReviewState
from .translation_evidence import (
    ConsistencyMemory,
    analyze_translation_evidence,
    build_consistency_memory,
    build_evidence_context,
)

logger = logging.getLogger(__name__)


def apply_translation_evidence_to_pages(pages: list[PreparedPage]) -> ConsistencyMemory:
    memory = build_consistency_memory(
        (block.text, TranslationReviewState.for_page(page).translation_for(block, ""))
        for page in pages
        for block in page.render_blocks
    )
    total_risks = 0
    total_repairs = 0
    for page in pages:
        apply_translation_evidence(
            page.render_blocks,
            page.translations,
            page.translation_contexts,
            memory,
        )
        total_risks += sum(1 for context in page.translation_contexts.values() if context.get("evidence_risk_flags"))
        total_repairs += sum(1 for context in page.translation_contexts.values() if context.get("evidence_repair_reasons"))
    logger.info(
        "Translation evidence prepared: pages=%d memory_terms=%d risk_blocks=%d repair_reason_blocks=%d",
        len(pages),
        len(memory.entries),
        total_risks,
        total_repairs,
    )
    return memory


def apply_translation_evidence(
    blocks: list[TextBlock],
    translations: dict[str, str],
    translation_contexts: dict[str, dict[str, object]],
    memory: ConsistencyMemory | None,
) -> None:
    state = TranslationReviewState(translations, translation_contexts)
    for block in blocks:
        translated = state.translation_for(block, "")
        context = state.context_for(block)
        context.update(build_evidence_context(block.text, translated, memory))
        state.update_line(block, context_updates=context)


def retry_cat_failures(page: PreparedPage, translator, config: PipelineConfig) -> tuple[int, int]:
    if not hasattr(translator, "retry_translation"):
        return (0, 0)
    from .translate import translation_debug_info

    attempted = 0
    accepted = 0
    suspect_attempted = 0
    suspect_accepted = 0
    state = TranslationReviewState.for_page(page)
    for block in page.render_blocks:
        context = state.context_for(block)
        previous_translation = state.translation_for(block, "")
        if context.get("cat_rejected") is not True and not should_retry_suspected_cat_translation(
            block.text,
            previous_translation,
            context,
            translator,
        ):
            continue
        attempted += 1
        previous_reason = str(context.get("cat_reject_reason") or "")
        if context.get("cat_rejected") is True:
            logger.info(
                "Retrying CAT rejected line: reason=%s source=%s",
                previous_reason,
                shorten(block.text),
            )
            retry_translation = translator.retry_translation(block.text)
            retry_mode = "rejected"
        else:
            suspect_attempted += 1
            logger.info(
                "Retrying suspected bad CAT line with strict second pass: translation=%s source=%s",
                shorten(previous_translation),
                shorten(block.text),
            )
            retry_translation = translator.second_retry_translation(block.text, previous_translation)
            retry_mode = "suspected_bad"
        retry_debug = _translator_debug_info_for(translator, block)
        context.update(
            {
                "cat_retry_attempted": True,
                "cat_retry_previous_translation": previous_translation,
                "cat_retry_previous_reject_reason": previous_reason,
                "cat_retry_mode": retry_mode,
                **retry_debug,
            }
        )
        if retry_debug.get("cat_retry_accepted") is True or retry_debug.get("cat_suspect_second_pass_accepted") is True:
            if retry_debug.get("cat_suspect_second_pass_accepted") is True:
                suspect_accepted += 1
            context.update(translation_debug_info(block.text, retry_translation, glossary_path=config.glossary_path))
            state.update_line(block, translated_text=retry_translation, context_updates=context)
            accepted += 1
            logger.info(
                "Accepted CAT retry translation: source=%s retry=%s",
                shorten(block.text),
                shorten(retry_translation),
            )
        else:
            logger.info(
                "Rejected CAT retry translation: source=%s reason=%s",
                shorten(block.text),
                retry_debug.get("cat_retry_reject_reason")
                or retry_debug.get("cat_suspect_second_pass_reject_reason")
                or "unknown",
            )
        if retry_debug.get("cat_retry_accepted") is not True and retry_debug.get("cat_suspect_second_pass_accepted") is not True:
            state.update_line(block, context_updates=context)
    if attempted:
        logger.info(
            "CAT retry pass finished for %s: attempted=%d accepted=%d suspect_attempted=%d suspect_accepted=%d",
            page.image_path,
            attempted,
            accepted,
            suspect_attempted,
            suspect_accepted,
        )
    return attempted, accepted


def should_retry_suspected_cat_translation(
    source_text: str,
    translated_text: str,
    context: dict[str, object],
    translator,
) -> bool:
    if not hasattr(translator, "second_retry_translation"):
        return False
    if context.get("cat_used") is not True:
        return False
    if context.get("cat_reason") == "phrasebook":
        return False
    if context.get("cat_retry_attempted") is True or context.get("cat_suspect_second_pass_attempted") is True:
        return False
    if context.get("cat_rejected") is True:
        return False
    if unusable_translation_reason(source_text, translated_text, translator_name="cat") is not None:
        return True
    return suspected_bad_translation(translated_text)


def apply_qwen_fallback_translations(
    blocks: list[TextBlock],
    translations: dict[str, str],
    translation_contexts: dict[str, dict[str, object]],
    page_order_report: list[dict[str, object]],
    qwen_fallback_translator,
    config: PipelineConfig,
    *,
    evidence_memory: ConsistencyMemory | None = None,
) -> tuple[int, int]:
    from .translate import translation_debug_info

    attempted = 0
    accepted = 0
    state = TranslationReviewState(translations, translation_contexts)
    for block in blocks:
        primary_translation = state.translation_for(block, "")
        primary_context = state.context_for(block)
        if not should_try_qwen_fallback(block.text, primary_translation, primary_context):
            continue

        attempted += 1
        context = context_for_block(block, page_order_report)
        logger.info(
            "Trying Qwen fallback model: box=%s source=%s primary=%s",
            block.box,
            shorten(block.text),
            shorten(primary_translation),
        )
        before_contexts = tuple(str(value) for value in context.get("context_before_window") or [])
        after_contexts = tuple(str(value) for value in context.get("context_after_window") or [])
        before_translations, after_translations = english_context_for_block(block, translations, page_order_report)
        guided_by_critic = primary_context.get("qwen_critic_should_repair") is True
        guided_by_evidence = bool(primary_context.get("evidence_repair_reasons"))
        use_guided_repair = (guided_by_critic or guided_by_evidence) and hasattr(
            qwen_fallback_translator,
            "repair_translation_with_guidance",
        )
        if use_guided_repair:
            fallback_translation = qwen_fallback_translator.repair_translation_with_guidance(
                block.text,
                current_translation=primary_translation,
                before=str(context.get("context_before") or "") or None,
                after=str(context.get("context_after") or "") or None,
                before_contexts=before_contexts,
                after_contexts=after_contexts,
                before_translations=before_translations,
                after_translations=after_translations,
                baseline=str(primary_context.get("qwen_baseline") or ""),
                critic_issues=tuple(str(value) for value in primary_context.get("qwen_critic_issues") or []),
                critic_reason=str(primary_context.get("qwen_critic_reason") or ""),
                visual_facts=visual_facts_for_context(primary_context),
                source_features=primary_context.get("source_features") if isinstance(primary_context.get("source_features"), dict) else None,
                translation_evidence=primary_context.get("translation_evidence") if isinstance(primary_context.get("translation_evidence"), dict) else None,
                consistency_memory=tuple(
                    entry for entry in primary_context.get("consistency_memory_entries", []) if isinstance(entry, dict)
                ),
                critic_source_evidence=tuple(str(value) for value in primary_context.get("qwen_critic_source_evidence") or []),
                critic_translation_evidence=tuple(str(value) for value in primary_context.get("qwen_critic_translation_evidence") or []),
            )
        else:
            fallback_translation = _translate_with_line_context(
                qwen_fallback_translator,
                block,
                block.text,
                before=str(context.get("context_before") or "") or None,
                after=str(context.get("context_after") or "") or None,
                before_contexts=before_contexts,
                after_contexts=after_contexts,
                visual_facts=visual_facts_for_context(primary_context),
            )
        fallback_debug = (
            _translator_debug_info_for(qwen_fallback_translator, block)
            if hasattr(qwen_fallback_translator, "debug_info_for")
            else {}
        )
        fallback_memory_entries = evidence_memory.entries_for_source(block.text, fallback_translation) if evidence_memory is not None else []
        fallback_evidence = analyze_translation_evidence(block.text, fallback_translation, memory_entries=fallback_memory_entries)
        reject_reason = fallback_reject_reason(block.text, fallback_translation, fallback_debug, fallback_evidence=fallback_evidence)
        primary_context.update(
            {
                "qwen_fallback_attempted": True,
                "qwen_fallback_source_translation": primary_translation,
                "qwen_fallback_translation": fallback_translation,
                "qwen_fallback_debug": fallback_debug,
                "qwen_fallback_accepted": reject_reason is None,
                "qwen_fallback_guided_by_critic": guided_by_critic,
                "qwen_fallback_guided_by_evidence": guided_by_evidence,
                "qwen_fallback_before_translations": list(before_translations),
                "qwen_fallback_after_translations": list(after_translations),
                "qwen_fallback_translation_evidence": fallback_evidence,
                "qwen_fallback_consistency_memory_entries": fallback_memory_entries,
                "cat_q8_fallback_attempted": primary_context.get("primary_translator") == "cat" or primary_context.get("cat_used") is True,
            }
        )
        if reject_reason is not None:
            primary_context["qwen_fallback_reject_reason"] = reject_reason
            state.update_line(block, context_updates=primary_context)
            logger.info(
                "Rejected Qwen fallback translation: reason=%s source=%s fallback=%s",
                reject_reason,
                shorten(block.text),
                shorten(fallback_translation),
            )
            continue

        primary_context.update(
            {
                "qwen_hybrid_used": True,
                "qwen_final_model": fallback_debug.get("qwen_model", "fallback"),
                "cat_q8_fallback_accepted": primary_context.get("primary_translator") == "cat" or primary_context.get("cat_used") is True,
                **translation_debug_info(block.text, fallback_translation, glossary_path=config.glossary_path),
            }
        )
        state.update_line(block, translated_text=fallback_translation, context_updates=primary_context)
        accepted += 1
        logger.info(
            "Accepted Qwen fallback translation: source=%s fallback=%s",
            shorten(block.text),
            shorten(fallback_translation),
        )

    if attempted:
        logger.info("Qwen fallback pass finished: attempted=%d accepted=%d", attempted, accepted)
    return attempted, accepted


def english_context_for_block(
    block: TextBlock,
    translations: dict[str, str],
    page_order_report: list[dict[str, object]],
    *,
    window: int = 3,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    target_index = None
    for index, item in enumerate(page_order_report):
        if str(item.get("line_id", "") or "") and str(item.get("line_id", "")) == str(block.metadata.get("line_id", "")):
            target_index = index
            break
        if tuple(item.get("box", ())) == tuple(block.box) and str(item.get("source_text", "")) == block.text:
            target_index = index
            break
    if target_index is None:
        return (), ()

    before: list[str] = []
    for item in page_order_report[max(0, target_index - window):target_index]:
        translated = translation_for_page_order_item(translations, item)
        if translated and translated != "...":
            before.append(translated)

    after: list[str] = []
    for item in page_order_report[target_index + 1:target_index + 1 + window]:
        translated = translation_for_page_order_item(translations, item)
        if translated and translated != "...":
            after.append(translated)

    return tuple(before), tuple(after)


def apply_qwen_critic_reviews(
    blocks: list[TextBlock],
    translations: dict[str, str],
    translation_contexts: dict[str, dict[str, object]],
    page_order_report: list[dict[str, object]],
    qwen_critic_translator,
    config: PipelineConfig,
    *,
    evidence_memory: ConsistencyMemory | None = None,
) -> tuple[int, int]:
    attempted = 0
    flagged = 0
    state = TranslationReviewState(translations, translation_contexts)
    for block in blocks:
        current_translation = state.translation_for(block, "")
        primary_context = state.context_for(block)
        if "source_features" not in primary_context or "translation_evidence" not in primary_context:
            primary_context.update(build_evidence_context(block.text, current_translation, evidence_memory))
            state.update_line(block, context_updates=primary_context)
        trigger_reasons = qwen_critic_trigger_reasons(block.text, current_translation, primary_context)
        if not trigger_reasons:
            continue

        attempted += 1
        context = context_for_block(block, page_order_report)
        logger.info(
            "Trying Qwen critic model: reasons=%s source=%s current=%s",
            ",".join(trigger_reasons),
            shorten(block.text),
            shorten(current_translation),
        )
        decision, critic_debug = qwen_critic_translator.critique_translation(
            block.text,
            current_translation=current_translation,
            before=str(context.get("context_before") or "") or None,
            after=str(context.get("context_after") or "") or None,
            before_contexts=tuple(str(value) for value in context.get("context_before_window") or []),
            after_contexts=tuple(str(value) for value in context.get("context_after_window") or []),
            baseline=str(primary_context.get("qwen_baseline") or ""),
            trigger_reasons=trigger_reasons,
            visual_facts=visual_facts_for_context(primary_context),
            source_features=primary_context.get("source_features") if isinstance(primary_context.get("source_features"), dict) else None,
            translation_evidence=primary_context.get("translation_evidence") if isinstance(primary_context.get("translation_evidence"), dict) else None,
            consistency_memory=tuple(
                entry for entry in primary_context.get("consistency_memory_entries", []) if isinstance(entry, dict)
            ),
        )
        repair_gate = qwen_critic_repair_decision(
            decision,
            source_text=block.text,
            translated_text=current_translation,
            translation_evidence=primary_context.get("translation_evidence") if isinstance(primary_context.get("translation_evidence"), dict) else None,
        )
        should_repair = bool(repair_gate["should_repair"])
        if should_repair:
            flagged += 1
        primary_context.update(
            {
                **critic_debug,
                "qwen_critic_flagged": should_repair,
                "qwen_critic_should_repair": should_repair,
                "qwen_critic_effective_issues": repair_gate["effective_issues"],
                "qwen_critic_repair_gate_reason": repair_gate["reason"],
                "qwen_critic_evidence_gate_reason": repair_gate["reason"],
            }
        )
        state.update_line(block, context_updates=primary_context)
        logger.info(
            "Qwen critic result: severity=%s issues=%s should_repair=%s source=%s reason=%s",
            decision.severity,
            ",".join(decision.issues),
            should_repair,
            shorten(block.text),
            shorten(decision.reason),
        )

    if attempted:
        logger.info("Qwen critic pass finished: attempted=%d flagged=%d", attempted, flagged)
    return attempted, flagged


def qwen_critic_trigger_reasons(
    source_text: str,
    translated_text: str,
    primary_context: dict[str, object],
) -> tuple[str, ...]:
    if primary_context.get("qwen_rejected") is True:
        return ()
    if primary_context.get("qwen_used") is not True and primary_context.get("cat_used") is not True:
        return ()
    if primary_context.get("qwen_reason") == "phrasebook" or primary_context.get("cat_reason") == "phrasebook":
        return ()
    if unusable_translation_reason(source_text, translated_text, translator_name="qwen") is not None:
        return ()
    if is_probable_standalone_japanese_name_or_credit(source_text):
        return ()
    if is_probable_short_sfx_or_reaction(source_text):
        return ()

    reasons: list[str] = ["accepted_line_review"]
    if suspected_bad_translation(translated_text):
        reasons.append("suspected_bad_translation")
    if primary_context.get("qwen_repair_used") is True:
        reasons.append("qwen_repair_used")
    if primary_context.get("qwen_unsupported_terms"):
        reasons.append("qwen_unsupported_terms")
    if primary_context.get("evidence_risk_flags"):
        reasons.append("evidence_risk")
    if primary_context.get("evidence_repair_reasons"):
        reasons.append("evidence_repair_reason")
    if has_critic_risk_source_terms(source_text):
        reasons.append("risk_source_terms")
    if has_awkward_english_pattern(translated_text):
        reasons.append("awkward_english_pattern")
    return tuple(dict.fromkeys(reasons))


def has_critic_risk_source_terms(source_text: str) -> bool:
    return "\u3008" in source_text


def is_probable_standalone_japanese_name_or_credit(source_text: str) -> bool:
    compact = re.sub(r"\s+", "", source_text)
    return bool(re.fullmatch(r"[\u4e00-\u9fff]{2,5}", compact))


def is_probable_short_sfx_or_reaction(source_text: str) -> bool:
    compact = re.sub(r"\s+", "", source_text)
    compact = compact.strip("\u300c\u300d\u300e\u300f")
    if len(compact) > 5:
        return False
    return bool(re.fullmatch(r"[\u3040-\u30ff\u31f0-\u31ff\u2026.!?\uff01\uff1f\u30fc]+", compact))


def has_awkward_english_pattern(translated_text: str) -> bool:
    normalized = translated_text.lower()
    patterns = (
        r"\b(?:father|mother|dad|mom|papa|mama)\s+and\s+(?:father|mother|dad|mom|papa|mama)\s+are\s+hated\b",
        r"\b[a-z]+ and [a-z]+ are hated\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def qwen_critic_should_trigger_repair(
    decision,
    *,
    source_text: str = "",
    translated_text: str = "",
    translation_evidence: dict[str, object] | None = None,
) -> bool:
    return bool(
        qwen_critic_repair_decision(
            decision,
            source_text=source_text,
            translated_text=translated_text,
            translation_evidence=translation_evidence,
        )["should_repair"]
    )


def qwen_critic_repair_decision(
    decision,
    *,
    source_text: str = "",
    translated_text: str = "",
    translation_evidence: dict[str, object] | None = None,
) -> dict[str, object]:
    if bool(getattr(decision, "ok", False)):
        return {"should_repair": False, "effective_issues": [], "reason": "critic_ok"}
    severity = str(getattr(decision, "severity", ""))
    issues = set(getattr(decision, "issues", ()) or ())
    hard_issues = {
        "context_mismatch",
        "omitted_term",
        "invented_detail",
        "name_drift",
        "untranslated_text",
        "glossary_conflict",
        "grammar_problem",
    }
    effective_issues = [
        issue
        for issue in sorted(issues & hard_issues)
        if critic_issue_has_local_repair_evidence(
            issue,
            source_text=source_text,
            translated_text=translated_text,
            translation_evidence=translation_evidence,
        )
    ]
    should_repair = bool(effective_issues)
    reason = "hard_issue_supported" if should_repair else "no_local_repair_evidence"
    if severity == "high" and effective_issues:
        reason = "high_severity_supported"
    return {
        "should_repair": should_repair,
        "effective_issues": effective_issues,
        "reason": reason,
    }


def critic_issue_has_local_repair_evidence(
    issue: str,
    *,
    source_text: str,
    translated_text: str,
    translation_evidence: dict[str, object] | None = None,
) -> bool:
    evidence_failures = set(str(value) for value in (translation_evidence or {}).get("preservation_failures", []))
    if issue == "omitted_term":
        return has_local_omission_evidence(source_text, translated_text) or bool(
            evidence_failures & {"dropped_number", "dropped_bracket_term"}
        )
    if issue == "invented_detail":
        return has_local_invention_evidence(translated_text) or "invented_english_name_on_noisy_source" in evidence_failures
    if issue == "name_drift":
        return has_local_name_drift_evidence(source_text, translated_text) or "consistency_conflict" in evidence_failures
    if issue == "untranslated_text":
        return has_local_untranslated_text_evidence(translated_text) or "untranslated_japanese" in evidence_failures
    if issue == "context_mismatch":
        return "consistency_conflict" in evidence_failures
    if issue == "grammar_problem":
        return "broken_english" in evidence_failures
    if issue in {"glossary_conflict", "grammar_problem"}:
        return True
    return False


def has_local_omission_evidence(source_text: str, translated_text: str) -> bool:
    if source_has_bracket_term(source_text) and not translation_preserves_bracket_term(source_text, translated_text):
        return True
    return False


def has_local_invention_evidence(translated_text: str) -> bool:
    return suspected_bad_translation(translated_text)


def has_local_name_drift_evidence(source_text: str, translated_text: str) -> bool:
    return False


def has_local_untranslated_text_evidence(translated_text: str) -> bool:
    if count_japanese_chars(translated_text) > 0:
        return True
    return bool(re.search(r"\b(?:wo|wa|ga|desu|kudasai|mite|shite|suru)\b", translated_text, flags=re.IGNORECASE))


def should_try_qwen_fallback(
    source_text: str,
    translated_text: str,
    primary_context: dict[str, object],
) -> bool:
    if primary_context.get("cat_rejected") is True:
        return True
    if primary_context.get("cat_bypassed") is True:
        return True
    if primary_context.get("qwen_critic_should_repair") is True:
        return True
    if primary_context.get("evidence_repair_reasons"):
        return True
    if primary_context.get("qwen_rejected") is True:
        return True
    if primary_context.get("qwen_repair_used") is True and primary_context.get("qwen_repair_accepted") is not True:
        return True
    if suspected_bad_translation(translated_text):
        return True
    return unusable_translation_reason(source_text, translated_text, translator_name="qwen") is not None


def fallback_reject_reason(
    source_text: str,
    translated_text: str,
    fallback_debug: dict[str, object],
    *,
    fallback_evidence: dict[str, object] | None = None,
) -> str | None:
    if fallback_debug.get("qwen_rejected") is True:
        return f"fallback_qwen_rejected:{fallback_debug.get('qwen_reject_reason', 'unknown')}"
    if suspected_bad_translation(translated_text):
        return "fallback_suspected_bad_translation"
    unusable_reason = unusable_translation_reason(source_text, translated_text, translator_name="qwen")
    if unusable_reason:
        return f"fallback_unusable:{unusable_reason}"
    evidence_failures = set(str(value) for value in (fallback_evidence or {}).get("preservation_failures", []))
    blocking_failures = {
        "dropped_number",
        "dropped_bracket_term",
        "invented_english_name_on_noisy_source",
        "broken_english",
        "untranslated_japanese",
    }
    blocking = sorted(evidence_failures & blocking_failures)
    if blocking:
        return f"fallback_evidence_failure:{','.join(blocking)}"
    return None


def visual_facts_for_context(context: dict[str, object]) -> tuple[str, ...]:
    if context.get("vision_facts_accepted") is not True:
        return ()
    facts: list[str] = []
    bubble_type = str(context.get("bubble_type") or "").strip()
    line_role = str(context.get("line_role") or "").strip()
    speaker_position = str(context.get("speaker_position") or "").strip()
    speaker_anchor = str(context.get("speaker_anchor") or "").strip()
    visible_emotion = str(context.get("visible_emotion") or "").strip()
    tone_hint = str(context.get("tone_hint") or "").strip()
    observable_action = str(context.get("observable_action") or "").strip()
    if bubble_type and bubble_type != "unknown":
        facts.append(f"Bubble type: {bubble_type}")
    if line_role and line_role != "unknown":
        facts.append(f"Line role: {line_role}")
    if speaker_position and speaker_position != "unknown":
        facts.append(f"Speaker position: {speaker_position}")
    if speaker_anchor and speaker_anchor != "unknown":
        facts.append(f"Speaker anchor: {speaker_anchor}")
    if visible_emotion and visible_emotion != "unknown":
        facts.append(f"Visible emotion: {visible_emotion}")
    if tone_hint and tone_hint != "unknown":
        facts.append(f"Tone hint: {tone_hint}")
    if observable_action and observable_action != "unknown":
        facts.append(f"Observable action: {observable_action}")
    for value in context.get("visual_facts") or []:
        text = str(value).strip()
        if text:
            facts.append(text)
    for value in context.get("context_hints") or []:
        text = str(value).strip()
        if text:
            facts.append(f"Context hint: {text}")
    return tuple(dict.fromkeys(facts))


def _translate_with_line_context(translator, block: TextBlock, text: str, **kwargs) -> str:
    try:
        return translator.translate_with_context(text, debug_id=translation_key_for_block(block), **kwargs)
    except TypeError:
        return translator.translate_with_context(text, **kwargs)


def _translator_debug_info_for(translator, block: TextBlock) -> dict[str, object]:
    try:
        return dict(translator.debug_info_for(block.text, debug_id=translation_key_for_block(block)))
    except TypeError:
        return dict(translator.debug_info_for(block.text))
