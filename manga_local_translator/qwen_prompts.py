from __future__ import annotations

from .translation_rules import normalize_japanese_for_translation


ANTI_HALLUCINATION_RULES = [
    "Anti-hallucination rules:",
    "- Translate only the provided target/source text.",
    "- Do not add dialogue, narration, sound effects, thoughts, names, facts, locations, relationships, motives, or backstory.",
    "- Do not invent a speaker name, gender, age, or relationship.",
    "- Use context only to resolve pronouns, tone, speaker continuity, and recurring terms.",
    "- If uncertain, become less specific, not more creative.",
    "- If OCR seems suspicious, keep close to the baseline instead of correcting it by guesswork.",
    "- Return JSON only.",
]


def build_qwen_translation_prompt(
    target: str,
    *,
    before: str | None,
    after: str | None,
    baseline: str,
    visual_facts: tuple[str, ...] = (),
) -> str:
    lines = [
        "/no_think",
        "Task: Improve the baseline translation of one Japanese manga bubble.",
        *ANTI_HALLUCINATION_RULES,
        "Preserve Japanese honorifics as romanized suffixes, for example Yamada san, Hime sama, Miki chan, Taro kun.",
        "Return only compact JSON: {\"translation\":\"...\",\"confidence\":0.0}",
    ]
    if before:
        lines.append(f"Previous bubble context: {normalize_japanese_for_translation(before)}")
    lines.append(f"Target Japanese bubble: {target}")
    if after:
        lines.append(f"Next bubble context: {normalize_japanese_for_translation(after)}")
    append_visual_facts(lines, visual_facts)
    lines.append(f"Baseline English translation: {baseline}")
    lines.append("JSON:")
    return "\n".join(lines)


def build_qwen_page_translation_prompt(
    page_items: list[dict[str, object]],
    *,
    previous_page_context: str | None,
) -> str:
    lines = [
        "/no_think",
        "Task: Translate one manga page from Japanese to concise English.",
        "Each numbered bubble must receive exactly one English translation.",
        *ANTI_HALLUCINATION_RULES,
        "Do not merge bubbles. Do not translate neighboring context into the current bubble.",
        "Preserve Japanese honorifics as romanized suffixes, for example Yamada san, Hime sama, Miki chan, Taro kun.",
        "Keep each translation short enough for a manga speech bubble.",
        "Return only compact JSON: {\"translations\":[{\"id\":1,\"translation\":\"...\",\"confidence\":0.0,\"reason\":\"...\"}]}",
    ]
    if previous_page_context:
        lines.append(f"Previous page final bubble context: {normalize_japanese_for_translation(previous_page_context)}")
    lines.append("Current page bubbles in manga reading order:")
    for item in page_items:
        page_id = int(item["id"])
        source = normalize_japanese_for_translation(str(item.get("source", "")))
        baseline = str(item.get("baseline", "")).strip()
        lines.append(f"{page_id}. Japanese: {source}")
        lines.append(f"   Baseline English: {baseline}")
    lines.append("JSON:")
    return "\n".join(lines)


def build_qwen_verification_prompt(
    target: str,
    *,
    before: str | None,
    after: str | None,
    before_contexts: tuple[str, ...] = (),
    after_contexts: tuple[str, ...] = (),
    baseline: str,
    candidate: str,
    visual_facts: tuple[str, ...] = (),
    ) -> str:
    lines = [
        "/no_think",
        "Task: Adjudicate two English translations for one Japanese manga bubble.",
        "Choose exactly one: candidate, baseline, or correction.",
        *ANTI_HALLUCINATION_RULES,
        "The target bubble is the only text being translated. Use the context window to check speaker continuity, pronouns, tone, and known terms.",
        "Reject the candidate if it adds unsupported nouns, events, numbers, places, jokes, relationships, or actions.",
        "Prefer baseline when the candidate is fluent but invents details or drifts away from the target bubble.",
        "Use correction only when a short, direct fix is obvious from the target Japanese.",
        "Honorifics should stay as romanized suffixes like san, sama, chan, kun, senpai, sensei, dono, or shi.",
        "Set ok=true only for choice candidate or correction. Set ok=false for choice baseline.",
        "In reason, say whether target text or context decided the choice.",
        "Keep reason under 14 words. Use at most 3 unsupported_terms.",
        "Return only one-line compact JSON: {\"choice\":\"candidate|baseline|correction\",\"ok\":true,\"translation\":\"...\",\"unsupported_terms\":[],\"reason\":\"...\"}",
    ]
    if before_contexts:
        lines.append("Earlier context window:")
        for index, value in enumerate(before_contexts, start=1):
            lines.append(f"  - {index}: {normalize_japanese_for_translation(value)}")
    if before:
        lines.append(f"Previous bubble context: {normalize_japanese_for_translation(before)}")
    lines.append(f"Target Japanese bubble: {target}")
    if after:
        lines.append(f"Next bubble context: {normalize_japanese_for_translation(after)}")
    if after_contexts:
        lines.append("Later context window:")
        for index, value in enumerate(after_contexts, start=1):
            lines.append(f"  - {index}: {normalize_japanese_for_translation(value)}")
    append_visual_facts(lines, visual_facts)
    lines.append(f"Baseline English translation: {baseline}")
    lines.append(f"Proposed English translation: {candidate}")
    lines.append("JSON:")
    return "\n".join(lines)


def build_qwen_critic_prompt(
    target: str,
    *,
    before: str | None,
    after: str | None,
    before_contexts: tuple[str, ...] = (),
    after_contexts: tuple[str, ...] = (),
    baseline: str,
    current_translation: str,
    trigger_reasons: tuple[str, ...] = (),
    visual_facts: tuple[str, ...] = (),
) -> str:
    lines = [
        "/no_think",
        "Task: Critique one accepted manga bubble translation. Do not translate or rewrite it.",
        *ANTI_HALLUCINATION_RULES,
        "Return issue labels only. Do not provide a replacement translation.",
        "Use severity none when the current translation is acceptable.",
        "Use severity low for minor style concerns that should not trigger repair.",
        "Use severity medium or high only when the line should be repaired by another model.",
        "For accepted_line_review, be conservative: do not flag ordinary wording preferences as medium/high.",
        "Use medium/high only for concrete translation failures: omission, invention, wrong name, context mismatch, untranslated text, glossary conflict, or broken grammar.",
        "Allowed issues: awkward_literal, context_mismatch, omitted_term, invented_detail, name_drift, tone_mismatch, untranslated_text, glossary_conflict, grammar_problem.",
        "Reject unsupported additions and missing target terms. Do not demand extra detail not present in Japanese.",
        "Return only compact JSON: {\"ok\":true,\"severity\":\"none|low|medium|high\",\"issues\":[],\"reason\":\"...\"}",
    ]
    if before_contexts:
        lines.append("Earlier context window:")
        for index, value in enumerate(before_contexts, start=1):
            lines.append(f"  - {index}: {normalize_japanese_for_translation(value)}")
    if before:
        lines.append(f"Previous bubble context: {normalize_japanese_for_translation(before)}")
    lines.append(f"Target Japanese bubble: {target}")
    if after:
        lines.append(f"Next bubble context: {normalize_japanese_for_translation(after)}")
    if after_contexts:
        lines.append("Later context window:")
        for index, value in enumerate(after_contexts, start=1):
            lines.append(f"  - {index}: {normalize_japanese_for_translation(value)}")
    append_visual_facts(lines, visual_facts)
    if trigger_reasons:
        lines.append(f"Why this line was selected for critique: {', '.join(trigger_reasons)}")
    lines.append(f"Baseline English translation: {baseline}")
    lines.append(f"Current accepted English translation: {current_translation}")
    lines.append("JSON:")
    return "\n".join(lines)


def build_qwen_repair_prompt(
    target: str,
    *,
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
) -> str:
    before_items = normalize_context_window(before_contexts, fallback=before)
    after_items = normalize_context_window(after_contexts, fallback=after)
    lines = [
        "/think",
        "Task: Repair one failed Japanese-to-English manga bubble translation.",
        *ANTI_HALLUCINATION_RULES,
        "Translate only TARGET. Do not translate neighboring context lines.",
        "Do not copy baseline or candidate if they contain hallucinations, garbage words, or unsupported details.",
        "If OCR is uncertain, give the shortest faithful English approximation.",
        "Keep the English short enough for a manga bubble, usually 16 words or fewer.",
        "Preserve Japanese honorifics as romanized suffixes like san, sama, chan, kun, senpai, sensei, dono, or shi.",
    ]
    if before_items:
        lines.append("Previous context, oldest to newest:")
        lines.extend(f"{index}. {value}" for index, value in enumerate(before_items, start=1))
    before_english = normalize_english_context_window(before_translations)
    if before_english:
        lines.append("Accepted English before target, oldest to newest:")
        lines.extend(f"{index}. {value}" for index, value in enumerate(before_english, start=1))
    lines.append(f"TARGET Japanese bubble: {target}")
    if after_items:
        lines.append("Next context, nearest to farthest:")
        lines.extend(f"{index}. {value}" for index, value in enumerate(after_items, start=1))
    after_english = normalize_english_context_window(after_translations)
    if after_english:
        lines.append("Accepted English after target, nearest to farthest:")
        lines.extend(f"{index}. {value}" for index, value in enumerate(after_english, start=1))
    append_visual_facts(lines, visual_facts)
    lines.append(f"Rejected baseline English: {baseline}")
    lines.append(f"Rejected candidate English: {candidate}")
    if reject_reason:
        lines.append(f"Why prior output failed: {reject_reason}")
    clean_critic_issues = [issue for issue in critic_issues if issue]
    if clean_critic_issues:
        lines.append(f"Critic issue labels to fix: {', '.join(clean_critic_issues)}")
    if critic_reason:
        lines.append(f"Critic reason: {critic_reason}")
    if clean_critic_issues or critic_reason:
        lines.append("Repair only the listed issue(s). Do not rewrite a correct line just for style.")
    lines.append("Return only one-line compact JSON: {\"translation\":\"...\",\"confidence\":0.0}")
    lines.append("JSON:")
    return "\n".join(lines)


def normalize_context_window(values: tuple[str, ...], *, fallback: str | None) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values or ((fallback,) if fallback else ()):
        text = normalize_japanese_for_translation(value)
        if not text or text in seen:
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def normalize_english_context_window(values: tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values:
        text = " ".join(str(value).split())
        if not text or text in seen or text == "...":
            continue
        seen.add(text)
        normalized.append(text)
    return normalized


def append_visual_facts(lines: list[str], visual_facts: tuple[str, ...]) -> None:
    clean = [str(value).strip() for value in visual_facts if str(value).strip()]
    if not clean:
        return
    lines.append("Visual facts from the page image:")
    lines.append("- These are context only. The target/source text remains authoritative.")
    lines.append("- Use them only for pronouns, speaker continuity, tone, emotion, and visible action.")
    lines.append("- Do not add facts to the translation unless the Japanese target text supports them.")
    for value in clean[:6]:
        lines.append(f"  - {value}")
