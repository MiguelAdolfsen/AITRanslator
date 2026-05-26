from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
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


def changed_files() -> str:
    try:
        completed = subprocess.run(["git", "status", "--porcelain"], text=True, encoding="utf-8", errors="replace", capture_output=True, check=False, timeout=10)
    except Exception:
        return "UNKNOWN"
    if completed.returncode != 0:
        return "UNKNOWN"
    paths = [line[3:].strip() for line in completed.stdout.splitlines() if len(line) > 3]
    return ";".join(paths)


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
        return {"schema_version": 1, "best_run_id": None, "best_score": None, "best_commit": None, "updated_at": None}
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


def maybe_update_best(best_path: Path, run_id: str, commit: str, metrics: dict[str, Any], *, disabled: bool, adapter_smoke: bool) -> bool:
    if disabled or adapter_smoke or not score.hard_guards_pass(metrics):
        return False
    best = load_best(best_path)
    current = best.get("best_score")
    run_score = float(metrics["source_extraction_score"])
    if current is not None and run_score >= float(current):
        return False
    best.update(
        {
            "schema_version": 1,
            "best_run_id": run_id,
            "best_score": run_score,
            "best_commit": commit,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "notes": "Updated by eval_source_extraction.py only when a run improves the score and passes hard guards.",
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
    parser.add_argument("--tests-ok", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--adapter-smoke", action="store_true")
    parser.add_argument("--strict-overlays", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--benchmark-set-name", default="")
    parser.add_argument("--baseline-score", type=float, default=0.0)
    parser.add_argument("--allow-external-output", action="store_true")
    parser.add_argument("--allow-external-results", action="store_true")
    parser.add_argument("--allow-external-benchmark", action="store_true")
    parser.add_argument("--no-update-best", action="store_true")
    args = parser.parse_args(argv)

    root = repo_root()
    loop_root = root / "source_extraction_autoresearch"
    ensure_under(args.output, loop_root / "runs", "--allow-external-output", args.allow_external_output)
    ensure_under(args.results, loop_root / "results", "--allow-external-results", args.allow_external_results)
    ensure_under(args.benchmark, loop_root, "--allow-external-benchmark", args.allow_external_benchmark)

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
    kept = maybe_update_best(
        args.results.parent / "best.json",
        args.run_id,
        commit,
        summary_metrics,
        disabled=args.no_update_best,
        adapter_smoke=args.adapter_smoke,
    )
    bench_name = benchmark_name(args.benchmark, args.benchmark_set_name or None)

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
        "tests_ok": "TRUE" if args.tests_ok else "FALSE",
        "fixture_validation_ok": "TRUE" if not validation_errors else "FALSE",
        "benchmark_set": bench_name,
        "source_extraction_score": summary_metrics["source_extraction_score"],
        "baseline_score": args.baseline_score,
        "delta_score": round(summary_metrics["source_extraction_score"] - args.baseline_score, 6) if args.baseline_score else 0,
        "kept": "TRUE" if kept else "FALSE",
        "notes": "adapter_smoke" if args.adapter_smoke else ("hard_failure" if summary_metrics["hard_failure"] else "ok"),
    }
    for key, value in summary_metrics.items():
        if key in RESULT_HEADER:
            row[key] = value
    append_result(args.results, row)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if summary_metrics["hard_failure"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

