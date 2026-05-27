from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

from manga_local_translator.config import PipelineConfig
from manga_local_translator.line_identity import lookup_context, lookup_translation, set_context
from manga_local_translator.page_cache import load_prepared_page_cache
from manga_local_translator.pipeline import (
    apply_qwen_critic_reviews,
    apply_qwen_fallback_translations,
    apply_translation_evidence_to_pages,
    release_qwen_translator,
    retry_cat_failures,
    translate_prepared_page,
    visual_facts_for_context,
)
from manga_local_translator.translate import build_translator
from manga_local_translator.vision_artifact import create_vision_artifact
from manga_local_translator.vision_prompt import VISION_FACTS_REQUIRED_FIELDS, build_vision_facts_json_repair_prompt
from manga_local_translator.vision_service import chunked, get_qwen_vision_client, vision_validation_score
from manga_local_translator.vision_types import VisionFactsRequest
from manga_local_translator.vision_validation import (
    parse_vision_facts_response,
    validate_vision_facts_results,
    vision_facts_acceptance_reason,
)

from .profiles import model_path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def load_benchmark(benchmark: Path) -> dict[str, Any]:
    manifest = json.loads((benchmark / "manifest.json").read_text(encoding="utf-8"))
    pages: list[dict[str, Any]] = []
    for path in sorted((benchmark / "labels").glob("*.references.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_label_path"] = str(path)
        pages.append(payload)
    references = read_jsonl(benchmark / "references.jsonl")
    split_by_page: dict[str, str] = {}
    splits = manifest.get("splits", {})
    for page_id in splits.get("main", []) or []:
        split_by_page[str(page_id)] = "main"
    for page_id in splits.get("holdout", []) or []:
        split_by_page[str(page_id)] = "holdout"
    return {
        "manifest": manifest,
        "pages": pages,
        "references": references,
        "references_by_line": {str(row["line_id"]): row for row in references},
        "split_by_page": split_by_page,
    }


def load_pages(page_metas: list[dict[str, Any]], *, output_dir: Path) -> list:
    pages = []
    for page in page_metas:
        page_id = str(page["page_id"])
        prepared_cache = Path(str(page["prepared_cache"]))
        pages.append(load_prepared_page_cache(prepared_cache, output_path=output_dir / f"{page_id}.png"))
    return pages


def primary_snapshot_from_pages(pages: list, page_metas: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for page, meta in zip(pages, page_metas):
        snapshot[str(meta["page_id"])] = {
            "translations": copy.deepcopy(page.translations),
            "translation_contexts": copy.deepcopy(page.translation_contexts),
        }
    return snapshot


def apply_primary_snapshot(pages: list, page_metas: list[dict[str, Any]], snapshot: dict[str, dict[str, Any]]) -> None:
    for page, meta in zip(pages, page_metas):
        payload = snapshot[str(meta["page_id"])]
        page.translations = copy.deepcopy(payload["translations"])
        page.translation_contexts = copy.deepcopy(payload["translation_contexts"])
        page.translation_fallback_blocks = []


def build_primary_snapshot(
    page_metas: list[dict[str, Any]],
    *,
    output_dir: Path,
    fake: bool,
    config: PipelineConfig,
    references_by_line: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    pages = load_pages(page_metas, output_dir=output_dir)
    if fake:
        for page in pages:
            for block in page.render_blocks:
                reference = references_by_line.get(str(block.metadata.get("line_id", "")), {})
                translated = str(reference.get("reference_text") or "...")
                from manga_local_translator.line_identity import set_translation

                set_translation(page.translations, block, translated)
                set_context(
                    page.translation_contexts,
                    block,
                    {
                        "primary_translator": "fake_cat",
                        "cat_used": True,
                        "cat_fake": True,
                        "translated_text": translated,
                    },
                )
        return primary_snapshot_from_pages(pages, page_metas)

    translator = build_translator("cat", cat_model_name=config.cat_model_name, glossary_path=config.glossary_path)
    try:
        for page in pages:
            translate_prepared_page(page, translator, config)
            retry_cat_failures(page, translator, config)
    finally:
        release_qwen_translator(translator)
    return primary_snapshot_from_pages(pages, page_metas)


def evaluate_profile(
    profile: dict[str, Any],
    page_metas: list[dict[str, Any]],
    primary_snapshot: dict[str, dict[str, Any]],
    *,
    output_dir: Path,
    fake: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pages = load_pages(page_metas, output_dir=output_dir)
    apply_primary_snapshot(pages, page_metas, primary_snapshot)
    timings: dict[str, float] = {}
    config = config_for_profile(profile)

    started = time.perf_counter()
    if bool(profile.get("vision_facts_enabled", True)):
        if fake:
            apply_fake_vision_facts(pages)
        else:
            apply_profile_vision_facts(pages, profile, config)
    timings["vision_seconds"] = round(time.perf_counter() - started, 3)

    evidence_memory = apply_translation_evidence_to_pages(pages)
    if not fake:
        started = time.perf_counter()
        critic = build_translator("qwen", qwen_model_path=model_path(profile, "qwen_critic_model"))
        try:
            for page in pages:
                apply_qwen_critic_reviews(
                    page.render_blocks,
                    page.translations,
                    page.translation_contexts,
                    page.page_order_report,
                    critic,
                    config,
                    evidence_memory=evidence_memory,
                )
        finally:
            release_qwen_translator(critic)
        timings["critic_seconds"] = round(time.perf_counter() - started, 3)

        started = time.perf_counter()
        fallback = build_translator("qwen", qwen_model_path=model_path(profile, "qwen_fallback_model"))
        try:
            for page in pages:
                apply_qwen_fallback_translations(
                    page.render_blocks,
                    page.translations,
                    page.translation_contexts,
                    page.page_order_report,
                    fallback,
                    config,
                    evidence_memory=evidence_memory,
                )
        finally:
            release_qwen_translator(fallback)
        timings["fallback_seconds"] = round(time.perf_counter() - started, 3)
    traces = traces_from_pages(pages, page_metas, profile)
    return traces, timings


def config_for_profile(profile: dict[str, Any]) -> PipelineConfig:
    return PipelineConfig(
        translator="cat",
        qwen_critic_model_path=model_path(profile, "qwen_critic_model"),
        qwen_fallback_model_path=model_path(profile, "qwen_fallback_model"),
        qwen_model_path=model_path(profile, "qwen_fallback_model"),
        vision_facts_enabled=bool(profile.get("vision_facts_enabled", True)),
        vision_mode=str(profile.get("artifact_mode") or "numbered_page"),
        vision_model_path=model_path(profile, "vision_model"),
        vision_projector_path=model_path(profile, "vision_projector"),
        debug=True,
        skip_render=True,
    )


def apply_fake_vision_facts(pages: list) -> None:
    for page in pages:
        for block in page.render_blocks:
            context = lookup_context(page.translation_contexts, block)
            context.update(
                {
                    "vision_facts_attempted": True,
                    "vision_facts_accepted": True,
                    "vision_facts_model": "fake_vision",
                    "bubble_type": "unknown",
                    "line_role": "unknown",
                    "speaker_position": "unknown",
                    "speaker_anchor": "unknown",
                    "visible_emotion": "unknown",
                    "tone_hint": "unknown",
                    "observable_action": "unknown",
                    "mapping_confidence": "medium",
                    "visual_facts": [],
                    "context_hints": [],
                }
            )
            set_context(page.translation_contexts, block, context)


def apply_profile_vision_facts(pages: list, profile: dict[str, Any], config: PipelineConfig) -> None:
    client = get_qwen_vision_client(config)
    for page in pages:
        artifact = create_vision_artifact(
            page.image_bgr,
            page.render_blocks,
            {},
            page.page_order_report,
            page.output_path,
            mode=config.vision_mode,
            artifact_label="vision-facts",
        )
        page.vision_facts_artifact = artifact
        requests = [
            VisionFactsRequest(
                line_id=entry.line_id,
                number=entry.number,
                source_text=entry.source_text,
                box=entry.box,
            )
            for entry in artifact.number_map
        ]
        entry_by_id = {entry.line_id: entry for entry in artifact.number_map}
        for request_chunk in chunked(requests, int(profile.get("chunk_size", 4) or 4)):
            prompt = build_profile_vision_prompt(request_chunk, profile, mode=config.vision_mode)
            image_path = vision_image_for_chunk(page, artifact.image_path, request_chunk, profile)
            raw_response = str(client.extract_facts(prompt, image_path))
            parsed = parse_vision_facts_response(raw_response)
            result_by_id, structural_errors = validate_vision_facts_results(request_chunk, parsed)
            json_repair_prompt = ""
            raw_json_repair_response = ""
            json_repair_used = False
            if structural_errors and hasattr(client, "repair_json"):
                json_repair_prompt = build_vision_facts_json_repair_prompt(request_chunk, raw_response)
                raw_json_repair_response = str(client.repair_json(json_repair_prompt) or "")
                repaired = parse_vision_facts_response(raw_json_repair_response)
                repaired_by_id, repaired_errors = validate_vision_facts_results(request_chunk, repaired)
                if vision_validation_score(repaired_by_id, repaired_errors) > vision_validation_score(result_by_id, structural_errors):
                    result_by_id = repaired_by_id
                    structural_errors = repaired_errors
                    json_repair_used = True
            for request in request_chunk:
                entry = entry_by_id[request.line_id]
                block = next((candidate for candidate in page.render_blocks if str(candidate.metadata.get("line_id", "")) == entry.line_id), None)
                if block is None:
                    continue
                result = result_by_id.get(request.line_id)
                reject_reason = vision_facts_acceptance_reason(
                    request,
                    result,
                    malformed_reason=structural_errors.get(request.line_id),
                )
                context = lookup_context(page.translation_contexts, block)
                accepted = reject_reason is None and result is not None
                context.update(
                    {
                        "vision_facts_attempted": True,
                        "vision_facts_accepted": accepted,
                        "vision_facts_reject_reason": reject_reason,
                        "vision_facts_structural_error": structural_errors.get(request.line_id),
                        "vision_facts_model": getattr(client, "model_name", "qwen_vision"),
                        "vision_facts_raw_prompt": prompt,
                        "vision_facts_raw_response": raw_response,
                        "vision_facts_image_path": str(image_path),
                        "vision_facts_raw_json_repair_prompt": json_repair_prompt,
                        "vision_facts_raw_json_repair_response": raw_json_repair_response,
                        "vision_facts_json_repair_used": json_repair_used,
                        "bubble_type": result.bubble_type if result else "unknown",
                        "line_role": result.line_role if result else "unknown",
                        "speaker_position": result.speaker_position if result else "unknown",
                        "speaker_anchor": result.speaker_anchor if result else "unknown",
                        "visible_emotion": result.visible_emotion if result else "unknown",
                        "tone_hint": result.tone_hint if result else "unknown",
                        "observable_action": result.observable_action if result else "unknown",
                        "mapping_confidence": result.mapping_confidence if result else "unknown",
                        "visual_facts": list(result.facts) if accepted else [],
                        "context_hints": list(result.context_hints) if accepted else [],
                        "vision_facts_risk_flags": list(result.risk_flags) if result else [],
                        "vision_facts_warnings": list(result.warnings) if result else [],
                    }
                )
                set_context(page.translation_contexts, block, context)


def build_profile_vision_prompt(requests: list[VisionFactsRequest], profile: dict[str, Any], *, mode: str) -> str:
    payload = [{"line_id": request.line_id, "number": request.number, "source_text": request.source_text} for request in requests]
    enabled_fields = [str(value) for value in profile.get("enabled_fields", []) or []]
    required_fields = [field for field in VISION_FACTS_REQUIRED_FIELDS if field in core_or_enabled(enabled_fields)]
    extra = profile.get("prompt", {}).get("extra_instructions", []) if isinstance(profile.get("prompt"), dict) else []
    lines = [
        "/no_think",
        "You are a conservative manga visual-context extractor.",
        "Task: describe only grounded visual facts for the numbered OCR lines listed below.",
        "Do not translate. Do not rewrite source_text. Do not OCR Japanese from the image.",
        "The provided source_text is authoritative and is included only for ID matching.",
        "Do not describe, paraphrase, quote, or translate the text printed in the bubble.",
        "Do not invent names, gender, relationships, motives, backstory, locations, dialogue, or narration.",
        "If uncertain, use unknown, set needs_review=true, and add a risk flag.",
        "Return JSON only. Do not use markdown fences. Do not include thinking text.",
        "For each input line, return exactly one object with the same line_id, number, and source_text copied exactly.",
        f"Required fields in every output line object: {', '.join(required_fields)}.",
        "Allowed bubble_type values: speech, thought, narration, sound_effect, sign, unknown.",
        "Allowed line_role values: speech, thought, narration, sound_effect, sign, metadata, unknown.",
        "Allowed mapping_confidence values: low, medium, high.",
        "speaker_anchor must be a visual anchor only, not a name.",
        "facts and context_hints must be short visual/context facts, not English translations.",
    ]
    lines.extend(str(value) for value in extra if str(value).strip())
    lines.extend(
        [
            "Output wrapper must be exactly: {\"lines\":[...]}",
            f"Vision mode: {mode}",
            f"Crop mode: {profile.get('crop_mode', 'numbered_full_page')}",
            "Lines to describe:",
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            "JSON:",
        ]
    )
    return "\n".join(lines)


def vision_image_for_chunk(page, artifact_path: Path, requests: list[VisionFactsRequest], profile: dict[str, Any]) -> Path:
    crop_mode = str(profile.get("crop_mode") or "numbered_full_page")
    if crop_mode == "numbered_full_page":
        return artifact_path
    padding = 48 if crop_mode == "line_crop" else 96
    crop_path = artifact_path.with_name(f"{artifact_path.stem}.{crop_mode}.{requests[0].number:03d}-{requests[-1].number:03d}.png")
    if crop_path.exists():
        return crop_path
    import cv2
    import numpy as np

    image = cv2.imread(str(artifact_path), cv2.IMREAD_COLOR)
    if image is None:
        return artifact_path
    x1, y1, x2, y2 = union_box([request.box for request in requests])
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(int(page.width), x2 + padding)
    y2 = min(int(page.height), y2 + padding)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return artifact_path
    if crop_mode == "full_page_plus_crop":
        full = image
        scale = min(1.0, max(240.0 / max(1, full.shape[1]), 240.0 / max(1, full.shape[0])))
        full_small = cv2.resize(full, (max(1, int(full.shape[1] * scale)), max(1, int(full.shape[0] * scale))))
        width = max(full_small.shape[1], crop.shape[1])
        full_canvas = pad_to_width(full_small, width)
        crop_canvas = pad_to_width(crop, width)
        separator = np.full((8, width, 3), 255, dtype=np.uint8)
        output = np.vstack([full_canvas, separator, crop_canvas])
    else:
        output = crop
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(crop_path), output):
        return artifact_path
    return crop_path


def union_box(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(int(box[0]) for box in boxes),
        min(int(box[1]) for box in boxes),
        max(int(box[2]) for box in boxes),
        max(int(box[3]) for box in boxes),
    )


def pad_to_width(image, width: int):
    import numpy as np

    if image.shape[1] >= width:
        return image
    pad = np.full((image.shape[0], width - image.shape[1], 3), 255, dtype=np.uint8)
    return np.hstack([image, pad])


def core_or_enabled(enabled_fields: list[str]) -> set[str]:
    return {
        "line_id",
        "number",
        "source_text",
        "mapping_confidence",
        "needs_review",
        "risk_flags",
        "warnings",
        *enabled_fields,
    }


def traces_from_pages(pages: list, page_metas: list[dict[str, Any]], profile: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page, meta in zip(pages, page_metas):
        labels_by_line = {str(label["line_id"]): label for label in meta["labels"]}
        for block in page.render_blocks:
            line_id = str(block.metadata.get("line_id", ""))
            label = labels_by_line.get(line_id, {})
            context = lookup_context(page.translation_contexts, block)
            final_translation = lookup_translation(page.translations, block, "")
            rows.append(
                {
                    "profile": profile.get("name", ""),
                    "profile_hash": profile.get("_profile_hash", ""),
                    "page_id": meta.get("page_id"),
                    "line_id": line_id,
                    "page_order": label.get("page_order", block.metadata.get("page_order")),
                    "source_text": block.text,
                    "source_box": list(block.box),
                    "source_hash": block.metadata.get("source_hash", ""),
                    "reference_text": label.get("reference_text", ""),
                    "reference_role": label.get("reference_role", ""),
                    "skip_reference": bool(label.get("skip_reference", False)),
                    "skip_reason": label.get("skip_reason", ""),
                    "primary_translation": context.get("cat_retry_previous_translation") or context.get("qwen_fallback_source_translation") or final_translation,
                    "final_translation": final_translation,
                    "visual_facts_used": list(visual_facts_for_context(context)),
                    "translation_context": context,
                }
            )
    return rows
