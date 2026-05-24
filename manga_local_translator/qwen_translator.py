from __future__ import annotations

import json
import logging
from pathlib import Path

from .hf_translators import OpusTranslator
from .logging_utils import shorten
from .qwen_ollama import ensure_ollama_model, find_ollama_executable, find_qwen_model_path, qwen_ollama_model_name, run_ollama_prompt
from .qwen_prompts import (
    build_qwen_critic_prompt,
    build_qwen_page_translation_prompt,
    build_qwen_repair_prompt,
    build_qwen_translation_prompt,
    build_qwen_verification_prompt,
)
from .qwen_types import (
    QWEN_CRITIC_SETTINGS,
    QWEN_REPAIR_SETTINGS,
    QWEN_PAGE_SETTINGS,
    QWEN_TRANSLATION_SETTINGS,
    QWEN_VERIFICATION_SETTINGS,
    QwenGenerationSettings,
    QwenCriticDecision,
    QwenVerificationDecision,
    qwen_settings_to_debug_dict,
)
from .qwen_validation import (
    accept_qwen_translation,
    parse_qwen_critic,
    parse_qwen_page_translations,
    parse_qwen_translation,
    parse_qwen_verification,
    should_try_qwen_repair,
    verifier_reason_says_reject,
)
from .translation_rules import (
    normalize_japanese_for_translation,
    postprocess_translation,
    prepare_source_for_translation,
    translate_known_phrase,
)
from .translator_base import Translator

logger = logging.getLogger(__name__)


class QwenTranslator(Translator):
    def __init__(
        self,
        model_path: Path | None = None,
        glossary_path: Path | None = None,
        n_ctx: int = 8192,
    ) -> None:
        super().__init__(glossary_path)
        self._model_path = model_path or find_qwen_model_path()
        self._ollama_model_name = qwen_ollama_model_name(self._model_path)
        logger.info("Initializing Qwen translator: model=%s", self._model_path)
        if not self._model_path.exists():
            raise RuntimeError(
                "Qwen GGUF model was not found. Put a .gguf file in .models/qwen, "
                "for example .models/qwen/Qwen3-8B-Q4_K_M.gguf"
            )
        self._cache: dict[str, str] = {}
        self._page_cache: dict[str, dict[int, str]] = {}
        self._debug: dict[str, dict[str, object]] = {}
        self._baseline_translator: OpusTranslator | None = None
        self._glossary_path = glossary_path
        self._llm = None
        self._ollama = find_ollama_executable()
        if self._ollama:
            try:
                self._ollama_model_name = ensure_ollama_model(
                    self._ollama,
                    self._ollama_model_name,
                    self._model_path,
                )
            except RuntimeError:
                logger.exception("Ollama is available but Qwen setup failed; trying llama-cpp-python fallback")
            else:
                self._backend = "ollama"
                logger.info("Qwen translator ready through Ollama model=%s", self._ollama_model_name)
                return

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "Qwen translator needs either Ollama or llama-cpp-python. "
                "Install Ollama, or run: pip install llama-cpp-python"
            ) from exc

        self._llm = Llama(
            model_path=str(self._model_path),
            n_ctx=n_ctx,
            n_gpu_layers=-1,
            verbose=False,
        )
        self._backend = "llama_cpp"
        logger.info("Qwen translator ready through llama-cpp-python")

    def translate(self, text: str) -> str:
        return self.translate_with_context(text)

    def translate_with_context(
        self,
        text: str,
        *,
        before: str | None = None,
        after: str | None = None,
        before_contexts: tuple[str, ...] = (),
        after_contexts: tuple[str, ...] = (),
        visual_facts: tuple[str, ...] = (),
    ) -> str:
        cache_key = json.dumps(
            {
                "text": text,
                "before": before or "",
                "after": after or "",
                "before_contexts": list(before_contexts),
                "after_contexts": list(after_contexts),
                "visual_facts": list(visual_facts),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if cache_key not in self._cache:
            translated, debug = self._translate_qwen(
                text,
                before=before,
                after=after,
                before_contexts=before_contexts,
                after_contexts=after_contexts,
                visual_facts=visual_facts,
            )
            self._cache[cache_key] = translated
            self._debug[normalize_japanese_for_translation(text)] = debug
        return self._cache[cache_key]

    def critique_translation(
        self,
        text: str,
        *,
        current_translation: str,
        before: str | None = None,
        after: str | None = None,
        before_contexts: tuple[str, ...] = (),
        after_contexts: tuple[str, ...] = (),
        baseline: str | None = None,
        trigger_reasons: tuple[str, ...] = (),
        visual_facts: tuple[str, ...] = (),
        source_features: dict[str, object] | None = None,
        translation_evidence: dict[str, object] | None = None,
        consistency_memory: tuple[dict[str, object], ...] = (),
    ) -> tuple[QwenCriticDecision, dict[str, object]]:
        normalized = normalize_japanese_for_translation(text)
        prepared, _replacements = prepare_source_for_translation(normalized, self._glossary)
        baseline_text = baseline if baseline is not None else self._baseline_translate(text)
        prompt = build_qwen_critic_prompt(
            prepared,
            before=before,
            after=after,
            before_contexts=before_contexts,
            after_contexts=after_contexts,
            baseline=baseline_text,
            current_translation=current_translation,
            trigger_reasons=trigger_reasons,
            visual_facts=visual_facts,
            source_features=source_features,
            translation_evidence=translation_evidence,
            consistency_memory=consistency_memory,
        )
        logger.debug("Qwen critic reviewing: source=%s current=%s", shorten(prepared), shorten(current_translation))
        raw = self._run_prompt(prompt, settings=QWEN_CRITIC_SETTINGS)
        decision = parse_qwen_critic(raw)
        debug = {
            "qwen_critic_attempted": True,
            "qwen_critic_model": self._ollama_model_name,
            "qwen_critic_backend": self._backend,
            "qwen_critic_settings": qwen_settings_to_debug_dict(QWEN_CRITIC_SETTINGS),
            "qwen_critic_trigger_reasons": list(trigger_reasons),
            "qwen_raw_critic_prompt": prompt,
            "qwen_raw_critic_response": raw,
            "qwen_critic_ok": decision.ok,
            "qwen_critic_severity": decision.severity,
            "qwen_critic_issues": list(decision.issues),
            "qwen_critic_reason": decision.reason,
            "qwen_critic_source_evidence": list(decision.source_evidence),
            "qwen_critic_translation_evidence": list(decision.translation_evidence),
            "qwen_critic_repair_recommended": decision.repair_recommended,
        }
        return decision, debug

    def repair_translation_with_guidance(
        self,
        text: str,
        *,
        current_translation: str,
        before: str | None = None,
        after: str | None = None,
        before_contexts: tuple[str, ...] = (),
        after_contexts: tuple[str, ...] = (),
        before_translations: tuple[str, ...] = (),
        after_translations: tuple[str, ...] = (),
        baseline: str | None = None,
        critic_issues: tuple[str, ...] = (),
        critic_reason: str | None = None,
        visual_facts: tuple[str, ...] = (),
        source_features: dict[str, object] | None = None,
        translation_evidence: dict[str, object] | None = None,
        consistency_memory: tuple[dict[str, object], ...] = (),
        critic_source_evidence: tuple[str, ...] = (),
        critic_translation_evidence: tuple[str, ...] = (),
    ) -> str:
        normalized = normalize_japanese_for_translation(text)
        prepared, _replacements = prepare_source_for_translation(normalized, self._glossary)
        baseline_text = baseline if baseline is not None else self._baseline_translate(text)
        reject_reason = "critic"
        if critic_issues:
            reject_reason = f"critic:{', '.join(critic_issues)}"
        elif translation_evidence and translation_evidence.get("preservation_failures"):
            reject_reason = f"evidence:{', '.join(str(value) for value in translation_evidence.get('preservation_failures', []))}"
        repair_candidate, raw_repair = self._repair_qwen_translation(
            source_text=prepared,
            before=before,
            after=after,
            before_contexts=before_contexts,
            after_contexts=after_contexts,
            before_translations=before_translations,
            after_translations=after_translations,
            baseline=baseline_text,
            candidate=current_translation,
            reject_reason=reject_reason,
            critic_issues=critic_issues,
            critic_reason=critic_reason,
            visual_facts=visual_facts,
            source_features=source_features,
            translation_evidence=translation_evidence,
            consistency_memory=consistency_memory,
            critic_source_evidence=critic_source_evidence,
            critic_translation_evidence=critic_translation_evidence,
        )
        debug: dict[str, object] = {
            "qwen_used": True,
            "qwen_model": self._ollama_model_name,
            "qwen_backend": self._backend,
            "qwen_guided_repair": True,
            "qwen_guided_repair_source_translation": current_translation,
            "qwen_guided_repair_critic_issues": list(critic_issues),
            "qwen_guided_repair_critic_reason": critic_reason or "",
            "qwen_guided_repair_critic_source_evidence": list(critic_source_evidence),
            "qwen_guided_repair_critic_translation_evidence": list(critic_translation_evidence),
            "qwen_guided_repair_before_translations": list(before_translations),
            "qwen_guided_repair_after_translations": list(after_translations),
            "qwen_repair_settings": qwen_settings_to_debug_dict(QWEN_REPAIR_SETTINGS),
            "qwen_baseline": baseline_text,
            "qwen_raw_repair": raw_repair,
            "qwen_repair_used": True,
            "qwen_repair_candidate": repair_candidate,
            "qwen_visual_facts_used": list(visual_facts),
        }
        accepted, reject = accept_qwen_translation(
            source_text=prepared,
            baseline=baseline_text,
            candidate=repair_candidate,
        )
        if accepted:
            verification = self._verify_qwen_translation(
                source_text=prepared,
                before=before,
                after=after,
                before_contexts=before_contexts,
                after_contexts=after_contexts,
                baseline=baseline_text,
                candidate=repair_candidate,
                visual_facts=visual_facts,
            )
            debug["qwen_repair_verify_choice"] = verification.choice
            debug["qwen_repair_verify_reason"] = verification.reason
            debug["qwen_raw_repair_verification"] = verification.raw_response
            debug["qwen_repair_verified_candidate"] = verification.translation
            verified_text = verification.translation or repair_candidate
            if verification.ok:
                verify_accepted, verify_reject = accept_qwen_translation(
                    source_text=prepared,
                    baseline=baseline_text,
                    candidate=verified_text,
                )
            else:
                verify_accepted = False
                verify_reject = f"repair_verification:{verification.choice}:{verification.reason}"
            if verify_accepted:
                result = postprocess_translation(normalized, prepared, verified_text, self._glossary)
                debug["qwen_repair_accepted"] = True
                debug["qwen_candidate"] = verified_text
                debug["qwen_final"] = result
                self._debug[normalized] = debug
                return result
            reject = verify_reject

        debug["qwen_repair_accepted"] = False
        debug["qwen_repair_reject_reason"] = reject
        debug["qwen_rejected"] = True
        debug["qwen_reject_reason"] = f"guided_repair_rejected:{reject}"
        debug["qwen_final"] = ""
        self._debug[normalized] = debug
        return ""

    def debug_info_for(self, text: str) -> dict[str, object]:
        return self._debug.get(normalize_japanese_for_translation(text), {})

    def translate_page(
        self,
        page_items: list[dict[str, object]],
        *,
        previous_page_context: str | None = None,
    ) -> dict[int, str]:
        cache_key = json.dumps(
            {
                "items": page_items,
                "previous_page_context": previous_page_context or "",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if cache_key in self._page_cache:
            return self._page_cache[cache_key]

        prepared_items: list[dict[str, object]] = []
        item_meta: dict[int, dict[str, object]] = {}
        translations: dict[int, str] = {}
        for item in page_items:
            page_id = int(item["id"])
            text = str(item.get("text", ""))
            normalized = normalize_japanese_for_translation(text)
            prepared, _replacements = prepare_source_for_translation(normalized, self._glossary)
            phrase = translate_known_phrase(normalized, self._glossary) or translate_known_phrase(prepared, self._glossary)
            baseline = self._baseline_translate(text)
            meta = {
                **item,
                "id": page_id,
                "text": text,
                "normalized": normalized,
                "prepared": prepared,
                "baseline": baseline,
            }
            item_meta[page_id] = meta
            if phrase is not None:
                translations[page_id] = phrase
                self._debug[normalized] = {
                    "qwen_used": False,
                    "qwen_page_used": False,
                    "qwen_page_id": page_id,
                    "qwen_reason": "phrasebook",
                    "qwen_model": self._ollama_model_name,
                }
                continue
            prepared_items.append({"id": page_id, "source": prepared, "baseline": baseline})

        if not prepared_items:
            self._page_cache[cache_key] = translations
            return translations

        prompt = build_qwen_page_translation_prompt(
            prepared_items,
            previous_page_context=previous_page_context,
        )
        logger.debug("Qwen page translation: items=%d previous=%s", len(prepared_items), shorten(previous_page_context or ""))
        raw_response = self._run_prompt(prompt, settings=QWEN_PAGE_SETTINGS)
        parsed_items = parse_qwen_page_translations(raw_response)
        duplicate_ids = duplicate_page_translation_ids(parsed_items)
        parsed_by_id = {
            item.page_id: item
            for item in parsed_items
            if item.page_id not in duplicate_ids
        }
        expected_ids = {int(item["id"]) for item in prepared_items}
        extra_ids = sorted(set(parsed_by_id) - expected_ids)

        for page_id in sorted(expected_ids):
            meta = item_meta[page_id]
            text = str(meta["text"])
            normalized = str(meta["normalized"])
            prepared = str(meta["prepared"])
            baseline = str(meta["baseline"])
            parsed = parsed_by_id.get(page_id)
            reject_reason = None
            candidate = ""
            original_candidate = ""
            verification = None
            if page_id in duplicate_ids:
                reject_reason = "duplicate_page_id"
            elif parsed is None:
                reject_reason = "missing_page_translation"
            else:
                candidate = parsed.translation
                original_candidate = candidate
                accepted, reason = accept_qwen_translation(
                    source_text=prepared,
                    baseline=baseline,
                    candidate=candidate,
                )
                if not accepted:
                    reject_reason = reason or "page_candidate_rejected"
                else:
                    verification = self._verify_qwen_translation(
                        source_text=prepared,
                        before=str(meta.get("before") or "") or None,
                        after=str(meta.get("after") or "") or None,
                        before_contexts=tuple(str(value) for value in meta.get("before_contexts") or []),
                        after_contexts=tuple(str(value) for value in meta.get("after_contexts") or []),
                        baseline=baseline,
                        candidate=candidate,
                    )
                    if verification.ok and verification.choice == "candidate" and verifier_reason_says_reject(verification.reason):
                        verification = QwenVerificationDecision(
                            ok=False,
                            choice=verification.choice,
                            reason=f"contradictory_verifier_reason:{verification.reason}",
                            translation=verification.translation,
                            unsupported_terms=verification.unsupported_terms,
                            raw_response=verification.raw_response,
                        )
                    verified_text = verification.translation or candidate
                    verified_accepted, verified_reject_reason = accept_qwen_translation(
                        source_text=prepared,
                        baseline=baseline,
                        candidate=verified_text,
                    )
                    if verification.ok and verified_accepted:
                        candidate = verified_text
                    else:
                        reject_reason = (
                            f"page_verification:{verification.choice}:{verification.reason}"
                            if not verification.ok
                            else f"page_verified_candidate_rejected:{verified_reject_reason}"
                        )

            page_debug = {
                "qwen_page_used": parsed is not None and reject_reason is None,
                "qwen_page_id": page_id,
                "qwen_page_reason": parsed.reason if parsed is not None else "",
                "qwen_page_confidence": parsed.confidence if parsed is not None else None,
                "qwen_page_candidate": candidate,
                "qwen_page_original_candidate": original_candidate,
                "qwen_page_rejected": reject_reason is not None,
                "qwen_page_reject_reason": reject_reason,
                "qwen_page_extra_ids": extra_ids,
                "qwen_page_duplicate_ids": sorted(duplicate_ids),
                "qwen_page_settings": qwen_settings_to_debug_dict(QWEN_PAGE_SETTINGS),
                "qwen_raw_page_prompt": prompt,
                "qwen_raw_page_response": raw_response,
            }
            if verification is not None:
                page_debug.update(
                    {
                        "qwen_page_verify_choice": verification.choice,
                        "qwen_page_verify_reason": verification.reason,
                        "qwen_raw_page_verification": verification.raw_response,
                        "qwen_page_verified_candidate": verification.translation,
                    }
                )

            if reject_reason is None:
                result = postprocess_translation(normalized, prepared, candidate, self._glossary)
                translations[page_id] = result
                self._debug[normalized] = {
                    "qwen_used": True,
                    "qwen_model": self._ollama_model_name,
                    "qwen_backend": self._backend,
                    "qwen_page_settings": qwen_settings_to_debug_dict(QWEN_PAGE_SETTINGS),
                    "qwen_baseline": baseline,
                    "qwen_candidate": candidate,
                    "qwen_final": result,
                    **page_debug,
                }
                continue

            fallback, debug = self._translate_qwen(
                text,
                before=str(meta.get("before") or "") or None,
                after=str(meta.get("after") or "") or None,
                before_contexts=tuple(str(value) for value in meta.get("before_contexts") or []),
                after_contexts=tuple(str(value) for value in meta.get("after_contexts") or []),
            )
            translations[page_id] = fallback
            debug.update(page_debug)
            self._debug[normalized] = debug

        self._page_cache[cache_key] = translations
        return translations

    def _translate_qwen(
        self,
        text: str,
        *,
        before: str | None,
        after: str | None,
        before_contexts: tuple[str, ...] = (),
        after_contexts: tuple[str, ...] = (),
        visual_facts: tuple[str, ...] = (),
    ) -> tuple[str, dict[str, object]]:
        normalized = normalize_japanese_for_translation(text)
        prepared, _replacements = prepare_source_for_translation(normalized, self._glossary)
        phrase = translate_known_phrase(normalized, self._glossary) or translate_known_phrase(prepared, self._glossary)
        if phrase is not None:
            return phrase, {
                "qwen_used": False,
                "qwen_reason": "phrasebook",
                "qwen_model": self._ollama_model_name,
            }

        baseline = self._baseline_translate(text)
        prompt = build_qwen_translation_prompt(prepared, before=before, after=after, baseline=baseline, visual_facts=visual_facts)
        logger.debug("Qwen translating: source=%s before=%s after=%s", shorten(prepared), shorten(before or ""), shorten(after or ""))
        raw_translation = self._run_prompt(prompt, settings=QWEN_TRANSLATION_SETTINGS)
        candidate = parse_qwen_translation(raw_translation)
        debug: dict[str, object] = {
            "qwen_used": True,
            "qwen_model": self._ollama_model_name,
            "qwen_backend": self._backend,
            "qwen_translation_settings": qwen_settings_to_debug_dict(QWEN_TRANSLATION_SETTINGS),
            "qwen_verification_settings": qwen_settings_to_debug_dict(QWEN_VERIFICATION_SETTINGS),
            "qwen_repair_settings": qwen_settings_to_debug_dict(QWEN_REPAIR_SETTINGS),
            "qwen_baseline": baseline,
            "qwen_raw_translation": raw_translation,
            "qwen_candidate": candidate,
            "qwen_visual_facts_used": list(visual_facts),
        }
        accepted, reason = accept_qwen_translation(
            source_text=prepared,
            baseline=baseline,
            candidate=candidate,
        )
        initially_accepted = accepted
        verification: QwenVerificationDecision | None = None
        if accepted or should_try_qwen_repair(reason):
            verification_candidate = candidate if accepted else f"[invalid candidate: {reason}]"
            verification = self._verify_qwen_translation(
                source_text=prepared,
                before=before,
                after=after,
                before_contexts=before_contexts,
                after_contexts=after_contexts,
                baseline=baseline,
                candidate=verification_candidate,
                visual_facts=visual_facts,
            )
            debug["qwen_verify_choice"] = verification.choice
            debug["qwen_verify_reason"] = verification.reason
            debug["qwen_raw_verification"] = verification.raw_response
            debug["qwen_verified_candidate"] = verification.translation
            if verification.unsupported_terms:
                debug["qwen_unsupported_terms"] = list(verification.unsupported_terms)
            if verification.ok and verification.choice == "candidate" and verifier_reason_says_reject(verification.reason):
                verification = QwenVerificationDecision(
                    ok=False,
                        choice=verification.choice,
                        reason=f"contradictory_verifier_reason:{verification.reason}",
                        translation=verification.translation,
                        unsupported_terms=verification.unsupported_terms,
                        raw_response=verification.raw_response,
                    )
                debug["qwen_verify_reason"] = verification.reason
            verified_text = verification.translation or candidate
            if verification.ok:
                verify_accepted, verify_filter_reason = accept_qwen_translation(
                    source_text=prepared,
                    baseline=baseline,
                    candidate=verified_text,
                )
                if verify_accepted:
                    candidate = verified_text
                    debug["qwen_candidate"] = candidate
                else:
                    verification = QwenVerificationDecision(
                        ok=False,
                        choice=verification.choice,
                        reason=f"verified_candidate_rejected:{verify_filter_reason}",
                        translation=verification.translation,
                        unsupported_terms=verification.unsupported_terms,
                        raw_response=verification.raw_response,
                    )
                    debug["qwen_verify_reason"] = verification.reason
            if not verification.ok:
                accepted = False
                reason = f"verification:{verification.choice}:{verification.reason}"
            elif verification.translation or not initially_accepted:
                accepted = True
                reason = None
        if not accepted:
            repair_candidate, raw_repair = self._repair_qwen_translation(
                source_text=prepared,
                before=before,
                after=after,
                before_contexts=before_contexts,
                after_contexts=after_contexts,
                baseline=baseline,
                candidate=candidate,
                reject_reason=reason,
                visual_facts=visual_facts,
            )
            debug["qwen_raw_repair"] = raw_repair
            debug["qwen_repair_used"] = True
            if repair_candidate:
                debug["qwen_repair_candidate"] = repair_candidate
                repair_accepted, repair_reject_reason = accept_qwen_translation(
                    source_text=prepared,
                    baseline=baseline,
                    candidate=repair_candidate,
                )
                if repair_accepted:
                    repair_verification = self._verify_qwen_translation(
                        source_text=prepared,
                        before=before,
                        after=after,
                            before_contexts=before_contexts,
                            after_contexts=after_contexts,
                            baseline=baseline,
                            candidate=repair_candidate,
                            visual_facts=visual_facts,
                        )
                    debug["qwen_repair_verify_choice"] = repair_verification.choice
                    debug["qwen_repair_verify_reason"] = repair_verification.reason
                    debug["qwen_raw_repair_verification"] = repair_verification.raw_response
                    debug["qwen_repair_verified_candidate"] = repair_verification.translation
                    repair_verified_text = repair_verification.translation or repair_candidate
                    if repair_verification.ok:
                        repair_verify_accepted, repair_verify_reject_reason = accept_qwen_translation(
                            source_text=prepared,
                            baseline=baseline,
                            candidate=repair_verified_text,
                        )
                    else:
                        repair_verify_accepted = False
                        repair_verify_reject_reason = f"repair_verification:{repair_verification.choice}:{repair_verification.reason}"
                    if repair_verify_accepted:
                        candidate = repair_verified_text
                        accepted = True
                        reason = None
                        debug["qwen_repair_accepted"] = True
                        debug["qwen_candidate"] = candidate
                    else:
                        debug["qwen_repair_accepted"] = False
                        debug["qwen_repair_reject_reason"] = repair_verify_reject_reason
                else:
                    debug["qwen_repair_accepted"] = False
                    debug["qwen_repair_reject_reason"] = repair_reject_reason
        if not accepted:
            baseline_accepted, baseline_reject_reason = accept_qwen_translation(
                source_text=prepared,
                baseline=baseline,
                candidate=baseline,
            )
            logger.info(
                "Rejecting Qwen translation: reason=%s source=%s baseline=%s candidate=%s baseline_reject_reason=%s",
                reason,
                shorten(prepared),
                shorten(baseline),
                shorten(candidate),
                baseline_reject_reason,
            )
            debug["qwen_rejected"] = True
            debug["qwen_reject_reason"] = reason
            if baseline_accepted:
                candidate = baseline
            else:
                debug["qwen_baseline_rejected"] = True
                debug["qwen_baseline_reject_reason"] = baseline_reject_reason
                candidate = ""
        result = postprocess_translation(normalized, prepared, candidate, self._glossary)
        debug["qwen_final"] = result
        return result, debug

    def _repair_qwen_translation(
        self,
        *,
        source_text: str,
        before: str | None,
        after: str | None,
        before_contexts: tuple[str, ...],
        after_contexts: tuple[str, ...],
        baseline: str,
        candidate: str,
        reject_reason: str | None,
        before_translations: tuple[str, ...] = (),
        after_translations: tuple[str, ...] = (),
        critic_issues: tuple[str, ...] = (),
        critic_reason: str | None = None,
        visual_facts: tuple[str, ...] = (),
        source_features: dict[str, object] | None = None,
        translation_evidence: dict[str, object] | None = None,
        consistency_memory: tuple[dict[str, object], ...] = (),
        critic_source_evidence: tuple[str, ...] = (),
        critic_translation_evidence: tuple[str, ...] = (),
    ) -> tuple[str, str]:
        prompt = build_qwen_repair_prompt(
            source_text,
            before=before,
            after=after,
            before_contexts=before_contexts,
            after_contexts=after_contexts,
            before_translations=before_translations,
            after_translations=after_translations,
            baseline=baseline,
            candidate=candidate,
            reject_reason=reject_reason,
            critic_issues=critic_issues,
            critic_reason=critic_reason,
            visual_facts=visual_facts,
            source_features=source_features,
            translation_evidence=translation_evidence,
            consistency_memory=consistency_memory,
            critic_source_evidence=critic_source_evidence,
            critic_translation_evidence=critic_translation_evidence,
        )
        raw = self._run_prompt(prompt, settings=QWEN_REPAIR_SETTINGS)
        return parse_qwen_translation(raw), raw

    def _verify_qwen_translation(
        self,
        *,
        source_text: str,
        before: str | None,
        after: str | None,
        before_contexts: tuple[str, ...] = (),
        after_contexts: tuple[str, ...] = (),
        baseline: str,
        candidate: str,
        visual_facts: tuple[str, ...] = (),
    ) -> QwenVerificationDecision:
        prompt = build_qwen_verification_prompt(
            source_text,
            before=before,
            after=after,
            before_contexts=before_contexts,
            after_contexts=after_contexts,
            baseline=baseline,
            candidate=candidate,
            visual_facts=visual_facts,
        )
        raw = self._run_prompt(prompt, settings=QWEN_VERIFICATION_SETTINGS)
        return parse_qwen_verification(raw)

    def _baseline_translate(self, text: str) -> str:
        if self._baseline_translator is None:
            self._baseline_translator = OpusTranslator(glossary_path=self._glossary_path)
        return self._baseline_translator.translate(text)

    def _run_prompt(self, prompt: str, *, settings: QwenGenerationSettings) -> str:
        if self._backend == "ollama":
            return run_ollama_prompt(
                self._ollama,
                self._ollama_model_name,
                prompt,
                settings=settings,
            )
        return self._llama_cpp_completion(prompt, settings=settings)

    def _llama_cpp_completion(self, prompt: str, *, settings: QwenGenerationSettings) -> str:
        if self._llm is None:
            return ""
        try:
            response = self._llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a professional Japanese-to-English manga dialogue translator. "
                            "Translate only the requested target line. Use context only to resolve pronouns, speakers, tone, and terms. "
                            "Return natural, concise English that can fit inside a manga speech bubble. "
                            "Do not explain. Do not include Japanese."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=settings.temperature,
                top_p=settings.top_p,
                top_k=settings.top_k,
                min_p=settings.min_p,
                max_tokens=settings.num_predict,
                stop=["\nJapanese:", "\nContext:", "\nTarget:"],
            )
            return str(response["choices"][0]["message"]["content"]).strip()
        except Exception:
            logger.exception("Qwen chat completion failed; trying plain completion")
            return self._plain_completion(prompt, settings=settings)

    def _plain_completion(self, prompt: str, *, settings: QwenGenerationSettings) -> str:
        completion = self._llm(
            f"Translate this Japanese manga dialogue to concise English.\n{prompt}\nEnglish:",
            temperature=settings.temperature,
            top_p=settings.top_p,
            top_k=settings.top_k,
            min_p=settings.min_p,
            max_tokens=settings.num_predict,
            stop=["\n", "Japanese:", "Context:", "Target:"],
        )
        return str(completion["choices"][0]["text"]).strip()


def duplicate_page_translation_ids(items) -> set[int]:
    seen: set[int] = set()
    duplicates: set[int] = set()
    for item in items:
        if item.page_id in seen:
            duplicates.add(item.page_id)
        seen.add(item.page_id)
    return duplicates
