from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import io_adapters
import score
from validate_fixtures import load_json, validate_benchmark
from visualize_overlays import draw_overlay


RESULT_HEADER = [
    "run_id",
    "timestamp",
    "commit",
    "parent_commit",
    "experiment_name",
    "changed_files",
    "tests_ok",
    "fixture_validation_ok",
    "benchmark_set",
    "source_extraction_score",
    "baseline_score",
    "delta_score",
    "pages_total",
    "gt_regions_total",
    "gt_extractable_regions",
    "predicted_regions",
    "matched_regions",
    "missed_dialogue_region_count",
    "false_positive_region_count",
    "destructive_false_positive_count",
    "harmless_false_positive_count",
    "mean_box_iou",
    "mean_ocr_cer",
    "severe_ocr_error_count",
    "wrong_text_region_match_count",
    "empty_ocr_count",
    "overmerge_count",
    "undermerge_count",
    "reading_order_error_count",
    "orientation_error_count",
    "bad_crop_count",
    "duplicate_region_count",
    "non_japanese_noise_count",
    "mean_extraction_ms_per_page",
    "p95_extraction_ms_per_page",
    "kept",
    "notes",
]

FORBIDDEN_DIRS = [
    "source_extraction_autoresearch/benchmarks/",
    "source_extraction_autoresearch/templates/",
    "render_autoresearch/",
    "translation_routing_autoresearch/",
]

FORBIDDEN_FILES = {
    "source_extraction_autoresearch/scripts/eval_source_extraction.py",
    "source_extraction_autoresearch/scripts/score.py",
    "source_extraction_autoresearch/scripts/match_regions.py",
    "source_extraction_autoresearch/scripts/normalize_text.py",
    "source_extraction_autoresearch/results/results.tsv",
    "manga_local_translator/render.py",
    "manga_local_translator/erase.py",
    "manga_local_translator/translate.py",
    "manga_local_translator/hf_translators.py",
    "manga_local_translator/qwen_translator.py",
    "manga_local_translator/qwen_validation.py",
    "manga_local_translator/qwen_vision.py",
    "manga_local_translator/vision_service.py",
    "manga_local_translator/vision_artifact.py",
}

FORBIDDEN_TEST_DIR = ".testing/tests/"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def git_value(args: list[str]) -> str:
    try:
        completed = subprocess.run(["git", *args], text=True, encoding="utf-8", errors="replace", capture_output=True, check=False, timeout=10)
    except Exception:
        return "UNKNOWN"
    return completed.stdout.strip() if completed.returncode == 0 and completed.stdout.strip() else "UNKNOWN"


def git_output(args: list[str], *, timeout: int = 10) -> dict[str, Any]:
    try:
        completed = subprocess.run(["git", *args], text=True, encoding="utf-8", errors="replace", capture_output=True, check=False, timeout=timeout)
    except Exception as exc:
        return {"returncode": -1, "stdout": "", "stderr": str(exc)}
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def changed_files() -> str:
    try:
        completed = subprocess.run(["git", "status", "--porcelain"], text=True, encoding="utf-8", errors="replace", capture_output=True, check=False, timeout=10)
    except Exception:
        return "UNKNOWN"
    if completed.returncode != 0:
        return "UNKNOWN"
    paths = [line[3:].strip() for line in completed.stdout.splitlines() if len(line) > 3]
    return ";".join(paths)


def changed_file_list() -> list[str]:
    value = changed_files()
    if not value or value == "UNKNOWN":
        return [] if not value else [value]
    return value.split(";")


def command_line() -> str:
    return subprocess.list2cmdline(sys.argv)


def normalize_repo_path(path: str) -> str:
    return path.replace("\\", "/").strip()


def is_evaluator_owned_path(path: str) -> bool:
    normalized = normalize_repo_path(path)
    if normalized.startswith("source_extraction_autoresearch/runs/"):
        return True
    if re.fullmatch(r"source_extraction_autoresearch/results/best\.[^/]+\.json", normalized):
        return True
    return False


def forbidden_dirty_files(paths: list[str], *, allow_dirty_results: bool) -> list[str]:
    forbidden: list[str] = []
    for path in paths:
        normalized = normalize_repo_path(path)
        if is_evaluator_owned_path(normalized):
            continue
        if normalized == "source_extraction_autoresearch/results/results.tsv" and allow_dirty_results:
            continue
        if normalized in FORBIDDEN_FILES:
            forbidden.append(normalized)
            continue
        if normalized.startswith(FORBIDDEN_TEST_DIR):
            forbidden.append(normalized)
            continue
        if any(normalized.startswith(prefix) for prefix in FORBIDDEN_DIRS):
            forbidden.append(normalized)
    return sorted(set(forbidden))


def run_tests(root: Path) -> dict[str, Any]:
    script = root / ".testing" / "run_tests.ps1"
    command = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)]
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    return {
        "command": subprocess.list2cmdline(command),
        "returncode": completed.returncode,
        "stdout_tail": "\n".join(completed.stdout.splitlines()[-80:]),
        "stderr_tail": "\n".join(completed.stderr.splitlines()[-80:]),
        "elapsed_seconds": round(elapsed, 3),
        "ok": completed.returncode == 0,
    }


def load_test_context(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def decision_preflight_errors(
    *,
    decision_benchmark: bool,
    adapter_smoke: bool,
    draft_benchmark: bool,
    skip_validation: bool,
    dirty_forbidden: list[str],
    allow_dirty_forbidden: bool,
    tests_available: bool,
    tests_ok: bool,
    allow_missing_tests: bool,
) -> list[str]:
    if not decision_benchmark:
        return []
    errors: list[str] = []
    if adapter_smoke:
        errors.append("decision benchmark runs cannot use --adapter-smoke")
    if draft_benchmark:
        errors.append("decision benchmark runs cannot target draft benchmarks")
    if skip_validation:
        errors.append("decision benchmark runs cannot use --skip-validation")
    if dirty_forbidden and not allow_dirty_forbidden:
        errors.append("decision benchmark run has dirty forbidden files:\n" + "\n".join(f"  - {path}" for path in dirty_forbidden))
    if not tests_available and not allow_missing_tests:
        errors.append("decision benchmark runs require --run-tests or --allow-missing-tests")
    if tests_available and not tests_ok:
        errors.append("decision benchmark tests did not pass")
    return errors


def repo_root() -> Path:
    return SCRIPT_DIR.parents[1]


def ensure_under(path: Path, parent: Path, flag_name: str, allowed: bool) -> None:
    resolved = path.resolve()
    parent_resolved = parent.resolve()
    if allowed:
        return
    if resolved == parent_resolved or parent_resolved in resolved.parents:
        return
    raise RuntimeError(f"{path} is outside {parent}; pass {flag_name} to allow it")


def ensure_results_header(results_path: Path) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    expected = "\t".join(RESULT_HEADER)
    if not results_path.exists() or results_path.stat().st_size == 0:
        results_path.write_text(expected + "\n", encoding="utf-8")
        return
    first = results_path.read_text(encoding="utf-8").splitlines()[0]
    if first != expected:
        raise RuntimeError(f"results header mismatch: {results_path}")


def append_result(results_path: Path, row: dict[str, Any]) -> None:
    ensure_results_header(results_path)
    needs_newline = results_path.read_bytes()[-1:] not in {b"\n", b"\r"}
    with results_path.open("a", encoding="utf-8", newline="") as handle:
        if needs_newline:
            handle.write("\n")
        writer = csv.DictWriter(handle, fieldnames=RESULT_HEADER, delimiter="\t", lineterminator="\n")
        writer.writerow({key: row.get(key, 0) for key in RESULT_HEADER})


def load_best(best_path: Path) -> dict[str, Any]:
    if not best_path.exists():
        return {
            "schema_version": 1,
            "benchmark_set": None,
            "best_run_id": None,
            "best_score": None,
            "best_commit": None,
            "updated_at": None,
        }
    return json.loads(best_path.read_text(encoding="utf-8-sig"))


def benchmark_fingerprint(benchmark: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(benchmark.rglob("*")):
        if path.is_file():
            rel = path.relative_to(benchmark).as_posix()
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def best_path_for_benchmark(results_dir: Path, benchmark_set: str) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", benchmark_set).strip("._")
    if not safe_name:
        safe_name = "unknown"
    return results_dir / f"best.{safe_name}.json"


def is_draft_benchmark(benchmark: Path, benchmark_set: str) -> bool:
    parts = [part.lower() for part in benchmark.parts]
    return "_draft" in benchmark_set.lower() or any("_draft" in part for part in parts)


def is_decision_benchmark(benchmark: Path, benchmark_set: str, *, adapter_smoke: bool) -> bool:
    return not adapter_smoke and not is_draft_benchmark(benchmark, benchmark_set)


def best_sort_key_from_metrics(metrics: dict[str, Any], *, changed_count: int) -> tuple[float, int, int, float, int, int, float, int]:
    merge_order_orientation = (
        int(metrics.get("overmerge_count", 0) or 0)
        + int(metrics.get("undermerge_count", 0) or 0)
        + int(metrics.get("reading_order_error_count", 0) or 0)
        + int(metrics.get("orientation_error_count", 0) or 0)
    )
    return (
        float(metrics.get("source_extraction_score", 0.0) or 0.0),
        int(metrics.get("missed_dialogue_region_count", 0) or 0),
        int(metrics.get("destructive_false_positive_count", 0) or 0),
        float(metrics.get("mean_ocr_cer", 0.0) or 0.0),
        int(metrics.get("severe_ocr_error_count", 0) or 0),
        merge_order_orientation,
        float(metrics.get("p95_extraction_ms_per_page", 0.0) or 0.0),
        changed_count,
    )


def best_sort_key(best: dict[str, Any]) -> tuple[float, int, int, float, int, int, float, int]:
    if "best_sort_key" in best:
        return tuple(best["best_sort_key"])  # type: ignore[return-value]
    score_value = float(best.get("best_score", 0.0) or 0.0)
    return (score_value, 0, 0, 0.0, 0, 0, 0.0, 0)


def maybe_update_best(
    best_path: Path,
    run_id: str,
    commit: str,
    benchmark_set: str,
    benchmark_hash: str,
    metrics: dict[str, Any],
    *,
    disabled: bool,
    decision_benchmark: bool,
    changed_count: int,
) -> bool:
    if disabled or not decision_benchmark or not score.hard_guards_pass(metrics):
        return False
    best = load_best(best_path)
    run_key = best_sort_key_from_metrics(metrics, changed_count=changed_count)
    if best.get("best_score") is not None and run_key >= best_sort_key(best):
        return False
    best.update(
        {
            "schema_version": 1,
            "benchmark_set": benchmark_set,
            "benchmark_fingerprint": benchmark_hash,
            "best_run_id": run_id,
            "best_score": float(metrics["source_extraction_score"]),
            "best_commit": commit,
            "best_sort_key": list(run_key),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "notes": "Per-benchmark best; smoke and draft benchmarks are excluded. Timing is a tie-breaker.",
        }
    )
    best_path.write_text(json.dumps(best, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True


def benchmark_name(benchmark: Path, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    parts = benchmark.parts
    if "cases" in parts:
        index = parts.index("cases")
        if index + 1 < len(parts):
            return f"cases/{parts[index + 1]}"
    return benchmark.name


def load_page_bundle(benchmark: Path, label_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    label = load_json(label_path)
    page_id = str(label["page_id"])
    ignore_path = benchmark / "ignore_regions" / f"{page_id}.ignore.json"
    config_path = benchmark / "configs" / f"{page_id}.config.json"
    ignore = load_json(ignore_path) if ignore_path.exists() else {"schema_version": 1, "page_id": page_id, "ignore_regions": []}
    config = load_json(config_path) if config_path.exists() else {"schema_version": 1, "page_id": page_id}
    return label, ignore, config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate source extraction against labeled page fixtures.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiment-name", default="")
    parser.add_argument("--tests-ok", action="store_true", help="Deprecated compatibility flag; decision runs require --run-tests or --allow-missing-tests.")
    parser.add_argument("--run-tests", action="store_true")
    parser.add_argument("--allow-missing-tests", action="store_true")
    parser.add_argument("--test-context", type=Path, default=None)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--adapter-smoke", action="store_true")
    parser.add_argument("--decision-run", dest="decision_run", action="store_true", default=None)
    parser.add_argument("--no-decision-run", dest="decision_run", action="store_false")
    parser.add_argument("--allow-dirty-forbidden", action="store_true")
    parser.add_argument("--allow-dirty-results", action="store_true")
    parser.add_argument("--strict-overlays", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--benchmark-set-name", default="")
    parser.add_argument("--baseline-score", type=float, default=0.0)
    parser.add_argument("--allow-external-output", action="store_true")
    parser.add_argument("--allow-external-results", action="store_true")
    parser.add_argument("--allow-external-benchmark", action="store_true")
    parser.add_argument("--no-update-best", action="store_true")
    args = parser.parse_args(argv)
    started_at = datetime.now(timezone.utc).isoformat()
    dirty_files_before = changed_file_list()
    test_context = load_test_context(args.test_context)

    root = repo_root()
    loop_root = root / "source_extraction_autoresearch"
    ensure_under(args.output, loop_root / "runs", "--allow-external-output", args.allow_external_output)
    ensure_under(args.results, loop_root / "results", "--allow-external-results", args.allow_external_results)
    ensure_under(args.benchmark, loop_root, "--allow-external-benchmark", args.allow_external_benchmark)

    bench_name = benchmark_name(args.benchmark, args.benchmark_set_name or None)
    auto_decision_benchmark = is_decision_benchmark(args.benchmark, bench_name, adapter_smoke=args.adapter_smoke)
    decision_benchmark = auto_decision_benchmark if args.decision_run is None else bool(args.decision_run)
    dirty_forbidden = forbidden_dirty_files(dirty_files_before, allow_dirty_results=args.allow_dirty_results)

    test_result: dict[str, Any] | None = None
    if args.run_tests:
        test_result = run_tests(root)
        if not test_result["ok"]:
            raise RuntimeError(f"test command failed before benchmark: {test_result['command']}")
    elif test_context is not None:
        test_result = test_context
    tests_ok = bool(test_result.get("ok")) if isinstance(test_result, dict) and "ok" in test_result else bool(args.tests_ok)
    preflight_errors = decision_preflight_errors(
        decision_benchmark=decision_benchmark,
        adapter_smoke=args.adapter_smoke,
        draft_benchmark=is_draft_benchmark(args.benchmark, bench_name),
        skip_validation=args.skip_validation,
        dirty_forbidden=dirty_forbidden,
        allow_dirty_forbidden=args.allow_dirty_forbidden,
        tests_available=test_result is not None,
        tests_ok=tests_ok,
        allow_missing_tests=args.allow_missing_tests,
    )
    if preflight_errors:
        raise RuntimeError("\n".join(preflight_errors))

    validation_errors: list[str] = []
    if not args.skip_validation:
        validation_errors = validate_benchmark(args.benchmark)
        if validation_errors:
            for error in validation_errors:
                print(error, file=sys.stderr)
            return 1

    if args.output.exists():
        if not args.overwrite_output:
            raise RuntimeError(f"output already exists: {args.output}; pass --overwrite-output")
        shutil.rmtree(args.output)
    (args.output / "overlays").mkdir(parents=True, exist_ok=True)

    per_page_metrics: list[dict[str, Any]] = []
    region_matches: list[dict[str, Any]] = []
    predicted_regions: list[dict[str, Any]] = []
    predicted_groups: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    adapter_notes: list[str] = []

    for label_path in sorted((args.benchmark / "labels").glob("*.labels.json")):
        label, ignore, config = load_page_bundle(args.benchmark, label_path)
        image_path = args.benchmark / str(label["image"])
        try:
            if args.adapter_smoke:
                prediction = io_adapters.extract_source_page_smoke(str(image_path), label)
            else:
                prediction = io_adapters.extract_source_page(str(image_path), config)
        except Exception as exc:
            failure = {"page_id": label.get("page_id"), "hard_failure": True, "error": str(exc)}
            failures.append(failure)
            raise

        adapter_notes.extend(str(item) for item in prediction.get("adapter_notes", []))
        page_metrics = score.score_page(label, prediction, ignore)
        per_page_metrics.append({key: value for key, value in page_metrics.items() if key != "matches"})
        region_matches.extend(page_metrics["matches"])
        failures.extend(page_metrics.get("false_positive_details", []))
        failures.extend(page_metrics.get("group_failure_details", []))
        failures.extend(page_metrics.get("reading_order_details", []))
        for region in prediction.get("regions", []):
            predicted_regions.append({"page_id": label["page_id"], **region})
        for group in prediction.get("groups", []):
            predicted_groups.append({"page_id": label["page_id"], **group})
        failures.extend([match for match in page_metrics["matches"] if match.get("violations")])
        try:
            draw_overlay(image_path, label, prediction, args.output / "overlays" / f"{label['page_id']}.overlay.png")
        except Exception as exc:
            if args.strict_overlays:
                raise
            adapter_notes.append(f"overlay failed for {label['page_id']}: {exc}")

    summary_metrics = score.summarize_pages(per_page_metrics)
    commit = git_value(["rev-parse", "HEAD"])
    parent_commit = git_value(["rev-parse", "HEAD^"])
    bench_hash = benchmark_fingerprint(args.benchmark)
    best_path = best_path_for_benchmark(args.results.parent, bench_name)
    current_best = load_best(best_path)
    baseline_score = args.baseline_score
    if decision_benchmark and not baseline_score and current_best.get("best_score") is not None:
        baseline_score = float(current_best["best_score"])
    kept = maybe_update_best(
        best_path,
        args.run_id,
        commit,
        bench_name,
        bench_hash,
        summary_metrics,
        disabled=args.no_update_best,
        decision_benchmark=decision_benchmark,
        changed_count=len(changed_file_list()),
    )

    write_jsonl(args.output / "per_page_metrics.jsonl", per_page_metrics)
    write_jsonl(args.output / "region_matches.jsonl", region_matches)
    write_jsonl(args.output / "predicted_regions.jsonl", predicted_regions)
    write_jsonl(args.output / "predicted_groups.jsonl", predicted_groups)
    write_jsonl(args.output / "failures.jsonl", failures)
    summary = {
        "schema_version": 1,
        "run_id": args.run_id,
        "benchmark_set": bench_name,
        "pages_total": summary_metrics["pages_total"],
        "source_extraction_score": summary_metrics["source_extraction_score"],
        "hard_failure": summary_metrics["hard_failure"],
        "metrics": summary_metrics,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": args.run_id,
                "benchmark": str(args.benchmark),
                "benchmark_fingerprint": bench_hash,
                "adapter_mode": "adapter-smoke" if args.adapter_smoke else "project",
                "decision_benchmark": decision_benchmark,
                "best_path": str(best_path),
                "adapter_notes": sorted(set(adapter_notes + io_adapters.ADAPTER_NOTES)),
                "fixture_validation_ok": not validation_errors,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    row = {
        "run_id": args.run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit": commit,
        "parent_commit": parent_commit,
        "experiment_name": args.experiment_name or args.run_id,
        "changed_files": changed_files(),
        "tests_ok": "TRUE" if tests_ok else "FALSE",
        "fixture_validation_ok": "TRUE" if not validation_errors else "FALSE",
        "benchmark_set": bench_name,
        "source_extraction_score": summary_metrics["source_extraction_score"],
        "baseline_score": baseline_score,
        "delta_score": round(summary_metrics["source_extraction_score"] - baseline_score, 6) if baseline_score else 0,
        "kept": "TRUE" if kept else "FALSE",
        "notes": (
            "adapter_smoke"
            if args.adapter_smoke
            else ("draft_benchmark" if not decision_benchmark else ("hard_failure" if summary_metrics["hard_failure"] else "ok"))
        ),
    }
    for key, value in summary_metrics.items():
        if key in RESULT_HEADER:
            row[key] = value
    append_result(args.results, row)
    completed_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1,
        "run_id": args.run_id,
        "started_at": started_at,
        "completed_at": completed_at,
        "command": command_line(),
        "argv": sys.argv,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "git": {
            "commit": commit,
            "parent_commit": parent_commit,
            "branch": git_value(["branch", "--show-current"]),
            "status_before": dirty_files_before,
            "status_after": changed_file_list(),
            "diff_name_status": git_output(["diff", "--name-status"]),
        },
        "benchmark": {
            "path": str(args.benchmark),
            "set": bench_name,
            "fingerprint": bench_hash,
            "decision_benchmark": decision_benchmark,
        },
        "results": {
            "path": str(args.results),
            "row": row,
            "best_path": str(best_path),
            "kept": kept,
            "baseline_score": baseline_score,
            "delta_score": row["delta_score"],
        },
        "adapter": {
            "mode": "adapter-smoke" if args.adapter_smoke else "project",
            "notes": sorted(set(adapter_notes + io_adapters.ADAPTER_NOTES)),
        },
        "validation": {
            "fixture_validation_ok": not validation_errors,
            "validation_skipped": args.skip_validation,
            "tests_ok": tests_ok,
            "test_result": test_result,
            "dirty_forbidden_before": dirty_forbidden,
            "allow_dirty_forbidden": args.allow_dirty_forbidden,
            "allow_dirty_results": args.allow_dirty_results,
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if summary_metrics["hard_failure"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
