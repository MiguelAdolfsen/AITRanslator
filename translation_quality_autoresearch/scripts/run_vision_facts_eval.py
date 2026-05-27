from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from . import io_adapters, profiles, scoring
    from .validate_benchmark import validate_benchmark
except ImportError:  # pragma: no cover
    from translation_quality_autoresearch.scripts import io_adapters, profiles, scoring
    from translation_quality_autoresearch.scripts.validate_benchmark import validate_benchmark


RESULT_HEADER = [
    "run_id",
    "timestamp",
    "commit",
    "benchmark",
    "baseline_profile",
    "baseline_profile_hash",
    "candidate_profile",
    "candidate_profile_hash",
    "main_quality_score",
    "main_baseline_quality_score",
    "main_delta",
    "main_critical_errors",
    "main_hallucinations",
    "holdout_quality_score",
    "holdout_baseline_quality_score",
    "holdout_delta",
    "holdout_pairwise_net",
    "promoted",
    "promotion_reason",
    "fake",
    "run_dir",
]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    benchmark = args.benchmark
    errors = validate_benchmark(benchmark)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    run_id = args.run_id or datetime.now().strftime("vision-facts-%Y%m%d-%H%M%S")
    run_dir = args.output_root / run_id
    artifacts_dir = run_dir / "artifacts"
    run_dir.mkdir(parents=True, exist_ok=False)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    baseline_profile = profiles.load_profile(args.baseline_profile)
    candidate_profile = profiles.load_profile(args.candidate_profile)
    baseline_profile["_profile_hash"] = profiles.profile_hash(baseline_profile)
    candidate_profile["_profile_hash"] = profiles.profile_hash(candidate_profile)

    benchmark_data = io_adapters.load_benchmark(benchmark)
    page_metas = filter_pages(benchmark_data["pages"], args.limit_pages)
    references_by_line = benchmark_data["references_by_line"]
    split_by_page = benchmark_data["split_by_page"]

    config = io_adapters.config_for_profile(candidate_profile)
    primary_snapshot = io_adapters.build_primary_snapshot(
        page_metas,
        output_dir=artifacts_dir / "primary",
        fake=args.fake,
        config=config,
        references_by_line=references_by_line,
    )
    baseline_traces, baseline_timings = io_adapters.evaluate_profile(
        baseline_profile,
        page_metas,
        primary_snapshot,
        output_dir=artifacts_dir / "baseline",
        fake=args.fake,
    )
    candidate_traces, candidate_timings = io_adapters.evaluate_profile(
        candidate_profile,
        page_metas,
        primary_snapshot,
        output_dir=artifacts_dir / "candidate",
        fake=args.fake,
    )

    judge = None
    if not args.fake:
        from manga_local_translator.translate import build_translator

        judge = build_translator("qwen", qwen_model_path=profiles.model_path(candidate_profile, "qwen_judge_model"))
    try:
        case_results = score_pairs(
            baseline_traces,
            candidate_traces,
            split_by_page=split_by_page,
            judge=judge,
            fake_judge=args.fake,
        )
    finally:
        if judge is not None:
            from manga_local_translator.pipeline import release_qwen_translator

            release_qwen_translator(judge)

    main_summary = scoring.summarize_case_results(case_results, split="main")
    holdout_summary = scoring.summarize_case_results(case_results, split="holdout")
    decision = scoring.promotion_decision(main_summary, holdout_summary)

    traces = [
        {"variant": "baseline", **row}
        for row in baseline_traces
    ] + [
        {"variant": "candidate", **row}
        for row in candidate_traces
    ]
    failures = [
        row
        for row in case_results
        if row.get("candidate_score", {}).get("deterministic_violations")
        or row.get("candidate_score", {}).get("critical_error_count", 0)
    ]
    io_adapters.write_jsonl(run_dir / "traces.jsonl", traces)
    io_adapters.write_jsonl(run_dir / "case_results.jsonl", case_results)
    io_adapters.write_jsonl(run_dir / "failures.jsonl", failures)

    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark": str(benchmark),
        "fake": bool(args.fake),
        "baseline_profile": profiles.public_profile(baseline_profile),
        "baseline_profile_hash": baseline_profile["_profile_hash"],
        "candidate_profile": profiles.public_profile(candidate_profile),
        "candidate_profile_hash": candidate_profile["_profile_hash"],
        "main": main_summary,
        "holdout": holdout_summary,
        "promotion": decision.__dict__,
        "timings": {
            "baseline": baseline_timings,
            "candidate": candidate_timings,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    append_results(args.results, result_row(summary, run_dir))

    if args.promote and decision.promoted and not args.fake:
        promote(candidate_profile, summary, args.results)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run paired vision-facts translation quality evaluations.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("translation_quality_autoresearch/benchmarks/frieren_ch26_pages_002_010"),
    )
    parser.add_argument("--baseline-profile", default="best")
    parser.add_argument("--candidate-profile", default="current_production")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--output-root", type=Path, default=Path("translation_quality_autoresearch/runs"))
    parser.add_argument("--results", type=Path, default=Path("translation_quality_autoresearch/results/translation_quality_results.tsv"))
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Plumbing smoke-test mode only: use fake CAT, fake vision facts, and heuristic judging. No CAT/Qwen/vision models are loaded.",
    )
    parser.add_argument("--promote", action="store_true", help="Commit and push candidate profile if promotion gates pass.")
    parser.add_argument("--limit-pages", default="", help="Comma-separated page IDs for smoke/debug runs.")
    return parser


def filter_pages(pages: list[dict[str, Any]], limit_pages: str) -> list[dict[str, Any]]:
    if not limit_pages.strip():
        return pages
    allowed = {value.strip() for value in limit_pages.split(",") if value.strip()}
    return [page for page in pages if str(page.get("page_id")) in allowed]


def score_pairs(
    baseline_traces: list[dict[str, Any]],
    candidate_traces: list[dict[str, Any]],
    *,
    split_by_page: dict[str, str],
    judge,
    fake_judge: bool,
) -> list[dict[str, Any]]:
    baseline_by_line = {str(row["line_id"]): row for row in baseline_traces}
    rows: list[dict[str, Any]] = []
    for candidate in candidate_traces:
        baseline = baseline_by_line[str(candidate["line_id"])]
        split = split_by_page.get(str(candidate["page_id"]), "main")
        baseline_score = scoring.score_trace(baseline, judge=judge, fake_judge=fake_judge)
        candidate_score = scoring.score_trace(candidate, judge=judge, fake_judge=fake_judge)
        baseline_payload = {**baseline, "score": baseline_score}
        candidate_payload = {**candidate, "score": candidate_score}
        pairwise = scoring.pairwise_compare(baseline_payload, candidate_payload, judge=judge, fake_judge=fake_judge)
        rows.append(
            {
                "page_id": candidate["page_id"],
                "line_id": candidate["line_id"],
                "split": split,
                "skip_reference": candidate["skip_reference"],
                "skip_reason": candidate["skip_reason"],
                "source_text": candidate["source_text"],
                "reference_text": candidate["reference_text"],
                "baseline_translation": baseline["final_translation"],
                "candidate_translation": candidate["final_translation"],
                "baseline_score": baseline_score,
                "candidate_score": candidate_score,
                "pairwise_winner": pairwise["winner"],
                "pairwise_reason": pairwise["reason"],
            }
        )
    return rows


def ensure_results_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\t".join(RESULT_HEADER)
    if not path.exists() or path.stat().st_size == 0:
        path.write_text(header + "\n", encoding="utf-8")
        return
    first = path.read_text(encoding="utf-8-sig").splitlines()[0]
    if first != header:
        raise RuntimeError(f"results header mismatch: {path}")


def append_results(path: Path, row: dict[str, Any]) -> None:
    ensure_results_header(path)
    needs_newline = path.read_bytes()[-1:] not in {b"\n", b"\r"}
    with path.open("a", encoding="utf-8", newline="") as handle:
        if needs_newline:
            handle.write("\n")
        writer = csv.DictWriter(handle, fieldnames=RESULT_HEADER, delimiter="\t", lineterminator="\n")
        writer.writerow({key: row.get(key, "") for key in RESULT_HEADER})


def result_row(summary: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    promotion = summary["promotion"]
    return {
        "run_id": summary["run_id"],
        "timestamp": summary["timestamp"],
        "commit": git_value(["rev-parse", "HEAD"]),
        "benchmark": summary["benchmark"],
        "baseline_profile": summary["baseline_profile"]["name"],
        "baseline_profile_hash": summary["baseline_profile_hash"],
        "candidate_profile": summary["candidate_profile"]["name"],
        "candidate_profile_hash": summary["candidate_profile_hash"],
        "main_quality_score": summary["main"]["quality_score"],
        "main_baseline_quality_score": summary["main"]["baseline_quality_score"],
        "main_delta": promotion["main_delta"],
        "main_critical_errors": summary["main"]["critical_error_count"],
        "main_hallucinations": summary["main"]["hallucination_count"],
        "holdout_quality_score": summary["holdout"]["quality_score"],
        "holdout_baseline_quality_score": summary["holdout"]["baseline_quality_score"],
        "holdout_delta": promotion["holdout_delta"],
        "holdout_pairwise_net": promotion["holdout_pairwise_net"],
        "promoted": promotion["promoted"],
        "promotion_reason": promotion["reason"],
        "fake": summary["fake"],
        "run_dir": str(run_dir),
    }


def promote(candidate_profile: dict[str, Any], summary: dict[str, Any], results_path: Path) -> None:
    best_profile_path = Path("translation_quality_autoresearch/profiles/best.json")
    best_result_path = Path("translation_quality_autoresearch/results/best.json")
    profiles.write_profile(best_profile_path, candidate_profile)
    best_result_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    controlled = [
        str(best_profile_path),
        str(best_result_path),
        str(results_path),
    ]
    subprocess.run(["git", "add", *controlled], check=True)
    message = f"Promote vision facts profile {candidate_profile.get('name')} ({summary['candidate_profile_hash']})"
    subprocess.run(["git", "commit", "-m", message], check=True)
    subprocess.run(["git", "push", "origin", "vision-facts-autoresearch"], check=True)


def git_value(args: list[str]) -> str:
    try:
        completed = subprocess.run(["git", *args], text=True, encoding="utf-8", errors="replace", capture_output=True, check=False, timeout=10)
    except Exception:
        return "UNKNOWN"
    if completed.returncode != 0:
        return "UNKNOWN"
    return completed.stdout.strip() or "UNKNOWN"


if __name__ == "__main__":
    raise SystemExit(main())
