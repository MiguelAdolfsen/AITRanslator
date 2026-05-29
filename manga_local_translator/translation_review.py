from __future__ import annotations

import re
from pathlib import Path

from .config import PipelineConfig
from .detect_types import TextBlock
from .line_identity import translation_for_page_order_item
from .page_types import PreparedPage
from .qwen_validation import source_has_bracket_term, translation_preserves_bracket_term
from .text_filter import count_japanese_chars, suspected_bad_translation, unusable_translation_reason
from .translated_state import TranslationReviewState
from .translation_evidence import (
    ConsistencyMemory,
    build_evidence_context,
)


def apply_translation_evidence_to_pages(pages: list[PreparedPage]) -> ConsistencyMemory:
    from .translation_review_page import TranslationReviewPage

    return TranslationReviewPage.apply_translation_evidence_to_pages(pages)


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


def _adapter_page(
    blocks: list[TextBlock],
    translations: dict[str, str],
    translation_contexts: dict[str, dict[str, object]],
    page_order_report: list[dict[str, object]],
):
    from .translation_review_page import TranslationReviewPage

    return TranslationReviewPage.for_page(
        PreparedPage(
            image_path=Path("translation-review-adapter"),
            output_path=Path("translation-review-adapter"),
            image_bgr=None,
            width=0,
            height=0,
            raw_blocks=blocks,
            render_blocks=blocks,
            skipped_blocks=[],
            grouping_report=[],
            page_order_report=page_order_report,
            translations=translations,
            translation_contexts=translation_contexts,
        )
    )


def retry_cat_failures(page: PreparedPage, translator, config: PipelineConfig) -> tuple[int, int]:
    from .translation_review_page import TranslationReviewPage

    return TranslationReviewPage.for_page(page).run_cat_retry(translator, config)


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
    return _adapter_page(blocks, translations, translation_contexts, page_order_report).run_qwen_fallback(
        qwen_fallback_translator,
        config,
        evidence_memory=evidence_memory,
    )


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
    return _adapter_page(blocks, translations, translation_contexts, page_order_report).run_qwen_critic(
        qwen_critic_translator,
        config,
        evidence_memory=evidence_memory,
    )


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
