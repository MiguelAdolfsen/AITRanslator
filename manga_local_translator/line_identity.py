from __future__ import annotations

import hashlib
from pathlib import Path

from .detect_types import TextBlock


def source_hash(source_text: str) -> str:
    return hashlib.sha256(source_text.encode("utf-8")).hexdigest()[:16]


def text_block_with_metadata(block: TextBlock, metadata: dict[str, object]) -> TextBlock:
    merged = dict(block.metadata)
    merged.update(metadata)
    merged.setdefault("source_hash", source_hash(block.text))
    return TextBlock(
        text=block.text,
        box=block.box,
        confidence=block.confidence,
        detector=block.detector,
        metadata=merged,
    )


def assign_ocr_block_ids(blocks: list[TextBlock], *, role: str = "ocr") -> list[TextBlock]:
    assigned: list[TextBlock] = []
    for index, block in enumerate(blocks, start=1):
        block_hash = stable_block_hash(block, index=index, role=role)
        assigned.append(
            text_block_with_metadata(
                block,
                {
                    "block_id": str(block.metadata.get("block_id") or f"{role}_{index:04d}_{block_hash}"),
                    "source_hash": source_hash(block.text),
                },
            )
        )
    return assigned


def assign_render_line_ids(
    blocks: list[TextBlock],
    page_order_report: list[dict[str, object]],
    output_path: Path,
) -> list[TextBlock]:
    assigned: list[TextBlock] = []
    for fallback_index, block in enumerate(blocks, start=1):
        order_index = page_order_for_identity(block, page_order_report) or fallback_index
        line_id = str(block.metadata.get("line_id") or f"{output_path.stem}_{order_index:03d}")
        block_hash = stable_block_hash(block, index=order_index, role="render")
        assigned.append(
            text_block_with_metadata(
                block,
                {
                    "line_id": line_id,
                    "block_id": str(block.metadata.get("block_id") or f"render_{order_index:04d}_{block_hash}"),
                    "source_hash": source_hash(block.text),
                    "page_order": order_index,
                },
            )
        )
    return assigned


def enrich_page_order_report(
    page_order_report: list[dict[str, object]],
    blocks: list[TextBlock],
) -> list[dict[str, object]]:
    by_box_text = {
        (tuple(block.box), block.text): block
        for block in blocks
    }
    enriched: list[dict[str, object]] = []
    for item in page_order_report:
        payload = dict(item)
        block = by_box_text.get((tuple(item.get("box", ())), str(item.get("source_text", ""))))
        if block is not None:
            payload["line_id"] = line_id_for_block(block)
            payload["block_id"] = block_id_for_block(block)
            payload["source_hash"] = source_hash(block.text)
        else:
            payload.setdefault("source_hash", source_hash(str(item.get("source_text", ""))))
        enriched.append(payload)
    return enriched


def enrich_grouping_report(
    grouping_report: list[dict[str, object]],
    blocks: list[TextBlock],
) -> list[dict[str, object]]:
    by_box_text = {
        (tuple(block.box), block.text): block
        for block in blocks
    }
    enriched: list[dict[str, object]] = []
    for group in grouping_report:
        payload = dict(group)
        block = by_box_text.get((tuple(group.get("box", ())), str(group.get("source_text", ""))))
        if block is not None:
            payload["line_id"] = line_id_for_block(block)
            payload["block_id"] = block_id_for_block(block)
            payload["source_hash"] = source_hash(block.text)
        else:
            payload.setdefault("source_hash", source_hash(str(group.get("source_text", ""))))
        members = []
        for member in payload.get("members", []):
            if not isinstance(member, dict):
                continue
            member_payload = dict(member)
            member_payload.setdefault("source_hash", source_hash(str(member_payload.get("source_text", ""))))
            members.append(member_payload)
        payload["members"] = members
        enriched.append(payload)
    return enriched


def stable_block_hash(block: TextBlock, *, index: int, role: str) -> str:
    payload = "|".join(
        [
            role,
            str(index),
            block.detector,
            ",".join(str(int(value)) for value in block.box),
            source_hash(block.text),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def line_id_for_block(block: TextBlock) -> str:
    return str(block.metadata.get("line_id") or "")


def block_id_for_block(block: TextBlock) -> str:
    return str(block.metadata.get("block_id") or "")


def translation_key_for_block(block: TextBlock) -> str:
    return line_id_for_block(block) or block.text


def lookup_translation(translations: dict[str, str], block: TextBlock, default: str = "") -> str:
    key = translation_key_for_block(block)
    if key in translations:
        return translations.get(key, default)
    return translations.get(block.text, default)


def set_translation(translations: dict[str, str], block: TextBlock, translated_text: str) -> None:
    translations[translation_key_for_block(block)] = translated_text


def lookup_context(
    translation_contexts: dict[str, dict[str, object]],
    block: TextBlock,
) -> dict[str, object]:
    key = translation_key_for_block(block)
    if key in translation_contexts:
        return dict(translation_contexts.get(key, {}))
    return dict(translation_contexts.get(block.text, {}))


def set_context(
    translation_contexts: dict[str, dict[str, object]],
    block: TextBlock,
    context: dict[str, object],
) -> None:
    payload = dict(context)
    payload.setdefault("line_id", line_id_for_block(block))
    payload.setdefault("block_id", block_id_for_block(block))
    payload.setdefault("source_hash", source_hash(block.text))
    translation_contexts[translation_key_for_block(block)] = payload


def page_order_for_identity(block: TextBlock, page_order_report: list[dict[str, object]]) -> int | None:
    line_id = line_id_for_block(block)
    for item in page_order_report:
        if line_id and str(item.get("line_id", "")) == line_id:
            return int(item["page_order"])
        if tuple(item.get("box", ())) == tuple(block.box) and str(item.get("source_text", "")) == block.text:
            return int(item["page_order"])
    return None


def translation_for_page_order_item(translations: dict[str, str], item: dict[str, object]) -> str:
    line_id = str(item.get("line_id", "") or "")
    if line_id and line_id in translations:
        return translations.get(line_id, "")
    return translations.get(str(item.get("source_text", "")), "")


def migrate_state_to_line_ids(
    blocks: list[TextBlock],
    translations: dict[str, str],
    translation_contexts: dict[str, dict[str, object]],
) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    migrated_translations: dict[str, str] = {}
    migrated_contexts: dict[str, dict[str, object]] = {}
    for block in blocks:
        key = translation_key_for_block(block)
        if key in translations:
            migrated_translations[key] = str(translations[key])
        elif block.text in translations:
            migrated_translations[key] = str(translations[block.text])
        if key in translation_contexts:
            context = dict(translation_contexts[key])
        elif block.text in translation_contexts:
            context = dict(translation_contexts[block.text])
        else:
            context = {}
        if context:
            set_context(migrated_contexts, block, context)
    return migrated_translations, migrated_contexts


def translations_by_source_for_compat(blocks: list[TextBlock], translations: dict[str, str]) -> dict[str, str]:
    return {
        block.text: lookup_translation(translations, block, "")
        for block in blocks
    }
