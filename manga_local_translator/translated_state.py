from __future__ import annotations

from .detect_types import TextBlock
from .line_identity import lookup_context, set_context, set_translation
from .page_types import PreparedPage
from .text_filter import block_to_debug_dict, fallback_translation


def apply_fallback_translation_state(
    page: PreparedPage,
    block: TextBlock,
    *,
    reason: str,
    source_translation: str,
    append_report: bool = True,
) -> str:
    translated = fallback_translation(block.text, reason=reason)
    set_translation(page.translations, block, translated)
    context = lookup_context(page.translation_contexts, block)
    context.update(
        {
            "fallback_used": True,
            "fallback_reason": reason,
            "fallback_source_translation": source_translation,
            "fallback_translation": translated,
        }
    )
    set_context(page.translation_contexts, block, context)
    if append_report:
        page.translation_fallback_blocks.append(
            block_to_debug_dict(block, status="fallback", reason=reason, translated_text=translated)
        )
    return translated
