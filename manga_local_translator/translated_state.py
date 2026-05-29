from __future__ import annotations

from collections.abc import Mapping

from .detect_types import TextBlock
from .line_identity import lookup_context, set_context, set_translation, translation_key_for_block
from .page_types import PreparedPage
from .text_filter import block_to_debug_dict, fallback_translation


def update_translated_line_state(
    translations: dict[str, str] | None,
    translation_contexts: dict[str, dict[str, object]],
    block: TextBlock,
    *,
    translated_text: str | None = None,
    context_updates: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if translated_text is not None:
        if translations is None:
            raise ValueError("translations is required when translated_text is provided")
        set_translation(translations, block, translated_text)
    context = lookup_context(translation_contexts, block)
    if context_updates:
        context.update(dict(context_updates))
    set_context(translation_contexts, block, context)
    return dict(translation_contexts[translation_key_for_block(block)])


def text_block_for_translation_state(
    *,
    source_text: str,
    box: tuple[int, int, int, int],
    state_key: str,
) -> TextBlock:
    return TextBlock(
        source_text,
        box,
        0.0,
        metadata={"line_id": state_key if state_key != source_text else ""},
    )


def apply_fallback_translation_state(
    page: PreparedPage,
    block: TextBlock,
    *,
    reason: str,
    source_translation: str,
    append_report: bool = True,
) -> str:
    translated = fallback_translation(block.text, reason=reason)
    update_translated_line_state(
        page.translations,
        page.translation_contexts,
        block,
        translated_text=translated,
        context_updates={
            "fallback_used": True,
            "fallback_reason": reason,
            "fallback_source_translation": source_translation,
            "fallback_translation": translated,
        },
    )
    if append_report:
        page.translation_fallback_blocks.append(
            block_to_debug_dict(block, status="fallback", reason=reason, translated_text=translated)
        )
    return translated
