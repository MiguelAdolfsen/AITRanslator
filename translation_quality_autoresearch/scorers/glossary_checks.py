from __future__ import annotations

from typing import Any


def glossary_terms_for_case(case, glossary: dict[str, Any] | None) -> list[dict[str, Any]]:
    glossary = glossary or {}
    raw_terms = glossary.get("terms") or []
    terms = [term for term in raw_terms if isinstance(term, dict)]
    requested = set(str(term) for term in getattr(case, "glossary_terms", []) or [])
    source = getattr(case, "source_text", "")
    selected: list[dict[str, Any]] = []
    for term in terms:
        term_id = str(term.get("id") or "")
        source_term = str(term.get("source") or "")
        target = str(term.get("target") or "")
        if not target:
            continue
        if term_id in requested or source_term in requested or target in requested or (source_term and source_term in source):
            selected.append(term)
    return selected


def glossary_violations(case, candidate_text: str, glossary: dict[str, Any] | None) -> list[str]:
    violations: list[str] = []
    text = str(candidate_text)
    for term in glossary_terms_for_case(case, glossary):
        target = str(term.get("target") or "")
        if bool(term.get("case_sensitive", False)):
            ok = target in text
        else:
            ok = target.lower() in text.lower()
        if not ok:
            violations.append(str(term.get("id") or term.get("source") or target))
    for value in getattr(case, "must_preserve", []) or []:
        required = str(value)
        if required and required.lower() not in text.lower():
            violations.append(f"must_preserve:{required}")
    return list(dict.fromkeys(violations))
