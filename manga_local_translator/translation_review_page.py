from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import PipelineConfig
from .grouping import context_for_block
from .line_identity import translation_key_for_block
from .logging_utils import shorten
from .page_types import PreparedPage
from .translated_state import TranslationReviewState
from .translation_evidence import (
    ConsistencyMemory,
    analyze_translation_evidence,
    build_consistency_memory,
    build_evidence_context,
)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TranslationReviewPage:
    page: PreparedPage

    @classmethod
    def for_page(cls, page: PreparedPage) -> TranslationReviewPage:
        return cls(page)

    @classmethod
    def apply_translation_evidence_to_pages(cls, pages: list[PreparedPage]) -> ConsistencyMemory:
        memory = build_consistency_memory(
            (block.text, TranslationReviewState.for_page(page).translation_for(block, ""))
            for page in pages
            for block in page.render_blocks
        )
        total_risks = 0
        total_repairs = 0
        for page in pages:
            cls.for_page(page).apply_translation_evidence(memory)
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

    def apply_translation_evidence(self, memory: ConsistencyMemory | None) -> None:
        state = TranslationReviewState.for_page(self.page)
        for block in self.page.render_blocks:
            translated = state.translation_for(block, "")
            context = state.context_for(block)
            context.update(build_evidence_context(block.text, translated, memory))
            state.update_line(block, context_updates=context)

    def run_qwen_fallback(
        self,
        qwen_fallback_translator,
        config: PipelineConfig,
        *,
        evidence_memory: ConsistencyMemory | None = None,
    ) -> tuple[int, int]:
        from .translate import translation_debug_info
        from .translation_review import (
            english_context_for_block,
            fallback_reject_reason,
            should_try_qwen_fallback,
            visual_facts_for_context,
        )

        attempted = 0
        accepted = 0
        state = TranslationReviewState.for_page(self.page)
        for block in self.page.render_blocks:
            primary_translation = state.translation_for(block, "")
            primary_context = state.context_for(block)
            if not should_try_qwen_fallback(block.text, primary_translation, primary_context):
                continue

            attempted += 1
            context = context_for_block(block, self.page.page_order_report)
            logger.info(
                "Trying Qwen fallback model: box=%s source=%s primary=%s",
                block.box,
                shorten(block.text),
                shorten(primary_translation),
            )
            before_contexts = tuple(str(value) for value in context.get("context_before_window") or [])
            after_contexts = tuple(str(value) for value in context.get("context_after_window") or [])
            before_translations, after_translations = english_context_for_block(
                block,
                self.page.translations,
                self.page.page_order_report,
            )
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

    def run_qwen_critic(
        self,
        qwen_critic_translator,
        config: PipelineConfig,
        *,
        evidence_memory: ConsistencyMemory | None = None,
    ) -> tuple[int, int]:
        from .translation_review import (
            qwen_critic_repair_decision,
            qwen_critic_trigger_reasons,
            visual_facts_for_context,
        )

        _ = config
        attempted = 0
        flagged = 0
        state = TranslationReviewState.for_page(self.page)
        for block in self.page.render_blocks:
            current_translation = state.translation_for(block, "")
            primary_context = state.context_for(block)
            if "source_features" not in primary_context or "translation_evidence" not in primary_context:
                primary_context.update(build_evidence_context(block.text, current_translation, evidence_memory))
                state.update_line(block, context_updates=primary_context)
            trigger_reasons = qwen_critic_trigger_reasons(block.text, current_translation, primary_context)
            if not trigger_reasons:
                continue

            attempted += 1
            context = context_for_block(block, self.page.page_order_report)
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

    def run_cat_retry(self, translator, config: PipelineConfig) -> tuple[int, int]:
        from .translate import translation_debug_info
        from .translation_review import should_retry_suspected_cat_translation

        if not hasattr(translator, "retry_translation"):
            return (0, 0)

        attempted = 0
        accepted = 0
        suspect_attempted = 0
        suspect_accepted = 0
        state = TranslationReviewState.for_page(self.page)
        for block in self.page.render_blocks:
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
                self.page.image_path,
                attempted,
                accepted,
                suspect_attempted,
                suspect_accepted,
            )
        return attempted, accepted


def _translate_with_line_context(translator, block, text: str, **kwargs) -> str:
    try:
        return translator.translate_with_context(text, debug_id=translation_key_for_block(block), **kwargs)
    except TypeError:
        return translator.translate_with_context(text, **kwargs)


def _translator_debug_info_for(translator, block) -> dict[str, object]:
    try:
        return dict(translator.debug_info_for(block.text, debug_id=translation_key_for_block(block)))
    except TypeError:
        return dict(translator.debug_info_for(block.text))
