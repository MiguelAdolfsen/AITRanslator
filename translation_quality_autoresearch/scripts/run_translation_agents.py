from __future__ import annotations

import argparse
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from translation_quality_autoresearch.common.io_utils import load_json, load_jsonl, load_jsonl_many, write_jsonl
from translation_quality_autoresearch.common.schemas import TranslationCase


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate live translation-agent traces for frozen source cases.")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--agents", default="opus,qwen_block,qwen_page,cat")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--glossary", type=Path)
    parser.add_argument("--config", type=Path, default=Path("translation_quality_autoresearch/config/default_live_agents.json"))
    parser.add_argument("--qwen-model", type=Path)
    parser.add_argument("--qwen-critic-model", type=Path)
    parser.add_argument("--qwen-fallback-model", type=Path)
    parser.add_argument("--cat-model")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--timeout-seconds", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict-agents", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_json(args.config, default={}) or {}
    cases = [TranslationCase.from_dict(row) for row in load_jsonl_many(args.cases)]
    if args.case_id:
        cases = [case for case in cases if case.case_id == args.case_id]
    if args.limit is not None:
        cases = cases[: args.limit]
    agents = [agent.strip() for agent in args.agents.split(",") if agent.strip()]
    existing_traces = load_jsonl(args.output) if args.resume and args.output.exists() else []
    existing_case_ids = {str(row.get("case_id") or "") for row in existing_traces}
    cases_to_run = [case for case in cases if case.case_id not in existing_case_ids]
    if args.dry_run:
        traces = existing_traces + [dry_trace(case, agents) for case in cases_to_run]
        write_jsonl(args.output, traces)
        print(f"dry-run wrote {len(traces)} trace records to {args.output} ({len(existing_traces)} resumed)")
        return 0

    translators: dict[str, Any] = {}
    unavailable: dict[str, str] = {}
    for agent in agents:
        try:
            translators[agent] = build_agent(agent, args)
        except Exception as exc:
            unavailable[agent] = str(exc)
            if args.strict_agents:
                raise
    traces = []
    for case in cases_to_run:
        trace = {
            "case_id": case.case_id,
            "source_hash": case.source_hash,
            "run_id": args.output.parent.name or "live_candidates",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "candidates": [],
        }
        for agent in agents:
            if agent in unavailable:
                trace["candidates"].append(skipped_candidate(case, agent, unavailable[agent]))
                continue
            trace["candidates"].append(run_agent(case, agent, translators[agent]))
        traces.append(trace)
    all_traces = existing_traces + traces
    write_jsonl(args.output, all_traces)
    print(f"wrote {len(all_traces)} trace records to {args.output} ({len(existing_traces)} resumed)")
    return 0


def build_agent(agent: str, args: argparse.Namespace):
    from manga_local_translator.translate import build_translator

    if agent == "opus":
        return build_translator("opus", glossary_path=args.glossary)
    if agent == "madlad":
        return build_translator("madlad", glossary_path=args.glossary)
    if agent == "argos":
        return build_translator("argos", glossary_path=args.glossary)
    if agent == "cat":
        return build_translator("cat", glossary_path=args.glossary, cat_model_name=args.cat_model)
    if agent in {"qwen_block", "qwen_page", "qwen_repair", "qwen_critic_repair", "qwen_vision_context_optional"}:
        return build_translator("qwen", glossary_path=args.glossary, qwen_model_path=args.qwen_model)
    raise RuntimeError(f"unsupported agent: {agent}")


def run_agent(case: TranslationCase, agent: str, translator) -> dict[str, Any]:
    started = time.perf_counter()
    raw = ""
    try:
        if agent == "qwen_page" and hasattr(translator, "translate_page"):
            page_result = translator.translate_page(
                [
                    {
                        "id": 1,
                        "text": case.source_text,
                        "before": first_or_none(case.context_before),
                        "after": first_or_none(case.context_after),
                        "before_contexts": case.context_before,
                        "after_contexts": case.context_after,
                        "line_id": case.case_id,
                    }
                ]
            )
            text = str(page_result.get(1, ""))
        elif hasattr(translator, "translate_with_context"):
            text = str(
                translator.translate_with_context(
                    case.source_text,
                    before=first_or_none(case.context_before),
                    after=first_or_none(case.context_after),
                    before_contexts=tuple(case.context_before),
                    after_contexts=tuple(case.context_after),
                    debug_id=case.case_id,
                )
            )
        else:
            text = str(translator.translate(case.source_text))
        debug = translator.debug_info_for(case.source_text, debug_id=case.case_id) if hasattr(translator, "debug_info_for") else {}
        raw = str(debug.get("qwen_raw_translation") or debug.get("qwen_raw_page_response") or debug.get("cat_raw_translation") or text)
        warnings: list[str] = []
    except Exception as exc:
        text = ""
        raw = str(exc)
        warnings = [f"agent_error:{exc}"]
    latency = int((time.perf_counter() - started) * 1000)
    prompt_name = "page_context_translation_prompt" if agent == "qwen_page" else "candidate_translation_prompt"
    prompt_hash = prompt_file_hash(prompt_name)
    return {
        "candidate_id": f"{case.case_id}__{agent}__001",
        "agent": agent,
        "text": text,
        "raw_output": raw,
        "prompt_name": prompt_name,
        "prompt_hash": prompt_hash,
        "model_name": str(getattr(translator, "_model_name", getattr(translator, "_ollama_model_name", agent))),
        "latency_ms": latency,
        "cost_proxy": cost_proxy(agent),
        "accepted_by_agent": not warnings and bool(text),
        "agent_warnings": warnings,
        "metadata": {},
    }


def skipped_candidate(case: TranslationCase, agent: str, reason: str) -> dict[str, Any]:
    return {
        "candidate_id": f"{case.case_id}__{agent}__skipped",
        "agent": agent,
        "text": "",
        "raw_output": reason,
        "prompt_name": None,
        "model_name": agent,
        "latency_ms": 0,
        "cost_proxy": cost_proxy(agent),
        "accepted_by_agent": False,
        "agent_warnings": [f"skipped_agent:{reason}"],
        "metadata": {"skipped": True},
    }


def dry_trace(case: TranslationCase, agents: list[str]) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "source_hash": case.source_hash,
        "run_id": "dry_run",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidates": [skipped_candidate(case, agent, "dry_run") for agent in agents],
    }


def prompt_file_hash(prompt_name: str) -> str:
    path = Path("translation_quality_autoresearch/prompts") / f"{prompt_name}.txt"
    if not path.exists():
        return ""
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def first_or_none(values: list[str]) -> str | None:
    return values[0] if values else None


def cost_proxy(agent: str) -> float:
    if agent in {"opus", "argos"}:
        return 1.0
    if agent in {"madlad", "cat"}:
        return 2.0
    return 4.0


if __name__ == "__main__":
    raise SystemExit(main())
