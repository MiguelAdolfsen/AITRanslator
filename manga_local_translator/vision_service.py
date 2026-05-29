from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from .config import PipelineConfig
from .debug_report import render_layout_warnings
from .detect_types import TextBlock
from .line_identity import translation_key_for_block
from .qwen_vision import QwenVisionClient
from .text_filter import count_japanese_chars, normalize_for_filter, suspected_bad_translation
from .translated_state import TranslationReviewState, text_block_for_translation_state
from .vision_artifact import create_vision_artifact
from .vision_prompt import (
    build_vision_facts_json_repair_prompt,
    build_vision_facts_prompt,
    build_vision_json_repair_prompt,
    build_vision_repair_prompt,
)
from .vision_types import VisionArtifact, VisionFactsRequest, VisionRepairRequest
from .vision_validation import (
    parse_vision_facts_response,
    parse_vision_repair_response,
    validate_vision_facts_results,
    validate_vision_results,
    vision_facts_acceptance_reason,
    vision_acceptance_reason,
)

logger = logging.getLogger(__name__)
_VISION_CLIENT_CACHE: dict[tuple[str, str], QwenVisionClient] = {}


class VisionClient(Protocol):
    model_name: str

    def repair(self, prompt: str, image_path: Path) -> str:
        ...

    def extract_facts(self, prompt: str, image_path: Path) -> str:
        ...


def apply_vision_repair(
    *,
    image_bgr,
    blocks: list[TextBlock],
    translations: dict[str, str],
    translation_contexts: dict[str, dict[str, object]],
    page_order_report: list[dict[str, object]],
    render_layouts,
    render_fits,
    output_path: Path,
    config: PipelineConfig,
    client: VisionClient | None = None,
) -> VisionArtifact | None:
    if not config.vision_enabled:
        return None
    state = TranslationReviewState(translations, translation_contexts)
    artifact = create_vision_artifact(
        image_bgr,
        blocks,
        translations,
        page_order_report,
        output_path,
        mode=config.vision_mode,
    )
    requests = build_vision_requests(
        blocks,
        translations,
        translation_contexts,
        artifact,
        render_layouts,
        render_fits,
        image_width=int(image_bgr.shape[1]),
        image_height=int(image_bgr.shape[0]),
        trigger=config.vision_trigger,
    )
    if not requests:
        logger.info("Vision repair skipped: no blocks matched trigger=%s", config.vision_trigger)
        return artifact

    prompt = build_vision_repair_prompt(requests, mode=config.vision_mode)
    if client is None:
        client = get_qwen_vision_client(config)
    raw_response = client.repair(prompt, artifact.image_path)
    parsed_results = parse_vision_repair_response(raw_response)
    result_by_id, structural_errors = validate_vision_results(requests, parsed_results)
    json_repair_prompt = ""
    raw_json_repair_response = ""
    json_repair_attempted = False
    json_repair_used = False
    repair_json = getattr(client, "repair_json", None)
    if structural_errors and callable(repair_json):
        json_repair_attempted = True
        json_repair_prompt = build_vision_json_repair_prompt(requests, raw_response)
        raw_json_repair_response = str(repair_json(json_repair_prompt) or "")
        repaired_results = parse_vision_repair_response(raw_json_repair_response)
        repaired_by_id, repaired_errors = validate_vision_results(requests, repaired_results)
        if vision_validation_score(repaired_by_id, repaired_errors) > vision_validation_score(result_by_id, structural_errors):
            result_by_id = repaired_by_id
            structural_errors = repaired_errors
            json_repair_used = True
            logger.info("Vision JSON repair improved response: errors=%d", len(structural_errors))
        else:
            logger.info("Vision JSON repair did not improve response: errors=%d", len(structural_errors))
    elif structural_errors:
        logger.info("Vision JSON repair unavailable: errors=%d", len(structural_errors))

    entry_by_id = {entry.line_id: entry for entry in artifact.number_map}
    accepted_count = 0
    for request in requests:
        entry = entry_by_id[request.line_id]
        result = result_by_id.get(request.line_id)
        reject_reason = vision_acceptance_reason(
            request,
            result,
            malformed_reason=structural_errors.get(request.line_id),
        )
        state_block = text_block_for_translation_state(
            source_text=entry.source_text,
            box=entry.box,
            state_key=entry.state_key or entry.source_text,
        )
        context_updates = {
            "vision_attempted": True,
            "line_id": entry.line_id,
            "block_id": entry.block_id,
            "source_hash": entry.source_hash,
            "vision_model": getattr(client, "model_name", "qwen_vision"),
            "vision_mode": config.vision_mode,
            "vision_trigger": config.vision_trigger,
            "vision_line_id": request.line_id,
            "vision_number": request.number,
            "vision_issue": request.issue,
            "vision_trigger_reason": request.issue,
            "vision_structural_error": structural_errors.get(request.line_id),
            "vision_translation": result.translation if result else "",
            "vision_source_text": result.source_text if result else "",
            "vision_accepted": reject_reason is None,
            "vision_reject_reason": reject_reason,
            "vision_final_source": "vision" if reject_reason is None else "text",
            "speaker": result.speaker if result else None,
            "speaker_confidence": result.speaker_confidence if result else None,
            "situation": result.situation if result else None,
            "visual_evidence_type": result.visual_evidence_type if result else "",
            "visual_evidence": result.visual_evidence if result else "",
            "needs_review": result.needs_review if result else False,
            "risk_flags": list(result.risk_flags) if result else [],
            "vision_warnings": list(result.warnings) if result else [],
            "vision_raw_prompt": prompt,
            "vision_raw_response": raw_response,
            "vision_json_repair_attempted": json_repair_attempted,
            "vision_json_repair_used": json_repair_used,
            "vision_raw_json_repair_prompt": json_repair_prompt,
            "vision_raw_json_repair_response": raw_json_repair_response,
        }
        state.update_line(
            state_block,
            translated_text=result.translation if reject_reason is None and result is not None else None,
            context_updates=context_updates,
        )
        if reject_reason is not None or result is None:
            continue
        accepted_count += 1
    logger.info("Vision repair finished: attempted=%d accepted=%d", len(requests), accepted_count)
    return artifact


def apply_vision_facts(
    *,
    image_bgr,
    blocks: list[TextBlock],
    translation_contexts: dict[str, dict[str, object]],
    page_order_report: list[dict[str, object]],
    output_path: Path,
    config: PipelineConfig,
    client: VisionClient | None = None,
) -> VisionArtifact | None:
    if not config.vision_facts_enabled or not blocks:
        return None
    state = TranslationReviewState({}, translation_contexts)
    artifact = create_vision_artifact(
        image_bgr,
        blocks,
        {},
        page_order_report,
        output_path,
        mode=config.vision_mode,
        artifact_label="vision-facts",
    )
    requests = [
        VisionFactsRequest(
            line_id=entry.line_id,
            number=entry.number,
            source_text=entry.source_text,
            box=entry.box,
        )
        for entry in artifact.number_map
    ]
    if not requests:
        logger.info("Vision facts skipped: no numbered text blocks")
        return artifact

    if client is None:
        client = get_qwen_vision_client(config)
    extract_facts = getattr(client, "extract_facts", None)
    entry_by_id = {entry.line_id: entry for entry in artifact.number_map}
    accepted_count = 0
    for request_chunk in chunked(requests, 4):
        prompt = build_vision_facts_prompt(request_chunk, mode=config.vision_mode)
        raw_response = str(extract_facts(prompt, artifact.image_path) if callable(extract_facts) else client.repair(prompt, artifact.image_path))
        parsed_results = parse_vision_facts_response(raw_response)
        result_by_id, structural_errors = validate_vision_facts_results(request_chunk, parsed_results)
        json_repair_prompt = ""
        raw_json_repair_response = ""
        json_repair_attempted = False
        json_repair_used = False
        repair_json = getattr(client, "repair_json", None)
        if structural_errors and callable(repair_json):
            json_repair_attempted = True
            json_repair_prompt = build_vision_facts_json_repair_prompt(request_chunk, raw_response)
            raw_json_repair_response = str(repair_json(json_repair_prompt) or "")
            repaired_results = parse_vision_facts_response(raw_json_repair_response)
            repaired_by_id, repaired_errors = validate_vision_facts_results(request_chunk, repaired_results)
            if vision_validation_score(repaired_by_id, repaired_errors) > vision_validation_score(result_by_id, structural_errors):
                result_by_id = repaired_by_id
                structural_errors = repaired_errors
                json_repair_used = True
                logger.info("Vision facts JSON repair improved response: errors=%d", len(structural_errors))

        for request in request_chunk:
            entry = entry_by_id[request.line_id]
            result = result_by_id.get(request.line_id)
            reject_reason = vision_facts_acceptance_reason(
                request,
                result,
                malformed_reason=structural_errors.get(request.line_id),
            )
            state_block = text_block_for_translation_state(
                source_text=entry.source_text,
                box=entry.box,
                state_key=entry.state_key or entry.source_text,
            )
            visual_facts = list(result.facts) if result and reject_reason is None else []
            context_hints = list(result.context_hints) if result and reject_reason is None else []
            context_updates = {
                "vision_facts_attempted": True,
                "line_id": entry.line_id,
                "block_id": entry.block_id,
                "source_hash": entry.source_hash,
                "vision_facts_model": getattr(client, "model_name", "qwen_vision"),
                "vision_facts_line_id": request.line_id,
                "vision_facts_number": request.number,
                "vision_facts_accepted": reject_reason is None,
                "vision_facts_reject_reason": reject_reason,
                "vision_facts_structural_error": structural_errors.get(request.line_id),
                "bubble_type": result.bubble_type if result else "unknown",
                "line_role": result.line_role if result else "unknown",
                "speaker_position": result.speaker_position if result else "unknown",
                "speaker_anchor": result.speaker_anchor if result else "unknown",
                "visible_emotion": result.visible_emotion if result else "unknown",
                "tone_hint": result.tone_hint if result else "unknown",
                "observable_action": result.observable_action if result else "unknown",
                "mapping_confidence": result.mapping_confidence if result else "unknown",
                "visual_facts": visual_facts,
                "context_hints": context_hints,
                "vision_facts_needs_review": result.needs_review if result else False,
                "vision_facts_risk_flags": list(result.risk_flags) if result else [],
                "vision_facts_warnings": list(result.warnings) if result else [],
                "vision_facts_json_repair_attempted": json_repair_attempted,
                "vision_facts_json_repair_used": json_repair_used,
            }
            if config.debug:
                context_updates.update(
                    {
                        "vision_facts_raw_prompt": prompt,
                        "vision_facts_raw_response": raw_response,
                        "vision_facts_raw_json_repair_prompt": json_repair_prompt,
                        "vision_facts_raw_json_repair_response": raw_json_repair_response,
                    }
                )
            state.update_line(
                state_block,
                context_updates=context_updates,
            )
            if reject_reason is None:
                accepted_count += 1
    logger.info("Vision facts finished: attempted=%d accepted=%d", len(requests), accepted_count)
    return artifact


def get_qwen_vision_client(config: PipelineConfig) -> QwenVisionClient:
    model_path = config.vision_model_path or config.qwen_model_path
    projector_path = config.vision_projector_path
    key = (str(model_path.resolve()) if model_path is not None else "", str(projector_path.resolve()) if projector_path is not None else "")
    client = _VISION_CLIENT_CACHE.get(key)
    if client is None:
        client = QwenVisionClient(model_path=model_path, projector_path=projector_path)
        _VISION_CLIENT_CACHE[key] = client
    return client


def release_qwen_vision_clients() -> None:
    if not _VISION_CLIENT_CACHE:
        return
    count = len(_VISION_CLIENT_CACHE)
    _VISION_CLIENT_CACHE.clear()
    try:
        import gc

        gc.collect()
    except Exception:
        logger.debug("Could not force garbage collection after releasing vision clients", exc_info=True)
    logger.info("Released %d Qwen vision client(s)", count)


def vision_validation_score(result_by_id: dict[str, object], structural_errors: dict[str, str]) -> int:
    return len(result_by_id) * 2 - len(structural_errors)


def chunked(values: list, size: int):
    for index in range(0, len(values), size):
        yield values[index : index + size]


def build_vision_requests(
    blocks: list[TextBlock],
    translations: dict[str, str],
    translation_contexts: dict[str, dict[str, object]],
    artifact: VisionArtifact,
    render_layouts,
    render_fits,
    *,
    image_width: int,
    image_height: int,
    trigger: str,
) -> list[VisionRepairRequest]:
    state = TranslationReviewState(translations, translation_contexts)
    entry_by_key = {entry.state_key or entry.source_text: entry for entry in artifact.number_map}
    entry_by_text = {entry.source_text: entry for entry in artifact.number_map}
    requests: list[VisionRepairRequest] = []
    for block, layout, fit in zip(blocks, render_layouts, render_fits):
        state_key = translation_key_for_block(block)
        entry = entry_by_key.get(state_key) or entry_by_text.get(block.text)
        if entry is None:
            continue
        context = state.context_for(block)
        translated = state.translation_for(block, "")
        if context.get("vision_attempted") is True:
            continue
        issue = vision_issue_for_block(
            block,
            translated,
            context,
            layout,
            fit,
            image_width=image_width,
            image_height=image_height,
            trigger=trigger,
        )
        if issue is None:
            continue
        requests.append(
            VisionRepairRequest(
                line_id=entry.line_id,
                number=entry.number,
                source_text=block.text,
                current_translation=translated,
                issue=issue,
                box=entry.box,
            )
        )
    return sorted(requests, key=lambda item: item.number)


def vision_issue_for_block(
    block: TextBlock,
    translated: str,
    context: dict[str, object],
    layout,
    fit,
    *,
    image_width: int,
    image_height: int,
    trigger: str,
) -> str | None:
    layout_warnings = render_layout_warnings(layout, fit, image_width=image_width, image_height=image_height)
    suspicious_reasons: list[str] = []
    if suspected_bad_translation(translated):
        suspicious_reasons.append("suspected_bad_translation")
    if context.get("qwen_rejected") is True:
        suspicious_reasons.append("qwen_rejected")
    if context.get("qwen_repair_used") is True and context.get("qwen_repair_accepted") is not True:
        suspicious_reasons.append("qwen_repair_failed")
    if context.get("qwen_fallback_attempted") is True and context.get("qwen_fallback_accepted") is not True:
        suspicious_reasons.append("qwen_fallback_rejected")
    if layout_warnings:
        suspicious_reasons.extend(f"layout:{warning}" for warning in layout_warnings)
    if is_low_confidence_short_ambiguous_block(block):
        suspicious_reasons.append("low_confidence_short_ocr")

    if trigger == "all":
        return "forced_all"
    if trigger == "layout":
        return ",".join(f"layout:{warning}" for warning in layout_warnings) if layout_warnings else None
    if suspicious_reasons:
        return ",".join(suspicious_reasons)
    return None


def is_low_confidence_short_ambiguous_block(block: TextBlock) -> bool:
    text = normalize_for_filter(block.text)
    if count_japanese_chars(text) == 0 or len(text) > 5:
        return False
    confidence = float(getattr(block, "confidence", 1.0) or 0.0)
    threshold = 0.45 if confidence <= 1.0 else 45.0
    return confidence <= threshold
