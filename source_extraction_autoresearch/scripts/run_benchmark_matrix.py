from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MATRIX = [
    ("synthetic", "synthetic", True, True),
    ("v2", "cases/local_real_diverse_v2_frozen", True, False),
    ("spy_v1", "cases/local_spy_short_v1_frozen", False, False),
    ("hard_v1", "cases/local_hard_pages_v1_frozen", False, False),
]


def run_command(command: list[str]) -> tuple[int, str, str, float]:
    print("running:", subprocess.list2cmdline(command))
    started = time.perf_counter()
    completed = subprocess.run(command, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)
    elapsed = time.perf_counter() - started
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    return completed.returncode, completed.stdout, completed.stderr, round(elapsed, 3)


def run_tests() -> dict[str, object]:
    command = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(Path(".testing/run_tests.ps1"))]
    returncode, stdout, stderr, elapsed = run_command(command)
    return {
        "command": subprocess.list2cmdline(command),
        "returncode": returncode,
        "stdout_tail": "\n".join(stdout.splitlines()[-80:]),
        "stderr_tail": "\n".join(stderr.splitlines()[-80:]),
        "elapsed_seconds": elapsed,
        "ok": returncode == 0,
    }


def load_summary(output_dir: Path) -> dict[str, object]:
    path = output_dir / "summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the source-extraction benchmark gate matrix.")
    parser.add_argument("--benchmark-root", type=Path, default=Path("source_extraction_autoresearch/benchmarks"))
    parser.add_argument("--output-root", type=Path, default=Path("source_extraction_autoresearch/runs"))
    parser.add_argument("--results", type=Path, default=Path("source_extraction_autoresearch/results/results.tsv"))
    parser.add_argument("--run-prefix", default="")
    parser.add_argument("--include-historical", action="store_true")
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument("--no-update-best", action="store_true")
    parser.add_argument("--allow-dirty-forbidden", action="store_true")
    parser.add_argument("--allow-dirty-results", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    args = parser.parse_args(argv)

    run_prefix = args.run_prefix or datetime.now(timezone.utc).strftime("matrix_%Y%m%d_%H%M%S")
    args.output_root.mkdir(parents=True, exist_ok=True)
    test_context = run_tests()
    test_context_path = args.output_root / f"{run_prefix}_test_context.json"
    test_context_path.write_text(json.dumps(test_context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not test_context.get("ok"):
        print("matrix tests failed; not running benchmarks")
        return 1

    matrix = [item for item in DEFAULT_MATRIX if item[2] or args.include_historical]
    failures: list[tuple[str, int, bool]] = []
    run_results: list[dict[str, object]] = []
    for short_name, rel_path, required, adapter_smoke in matrix:
        run_id = f"{run_prefix}_{short_name}"
        output_dir = args.output_root / run_id
        command = [
            sys.executable,
            str(Path(__file__).with_name("eval_source_extraction.py")),
            "--benchmark",
            str(args.benchmark_root / rel_path),
            "--output",
            str(output_dir),
            "--results",
            str(args.results),
            "--run-id",
            run_id,
            "--allow-missing-tests",
            "--test-context",
            str(test_context_path),
        ]
        if adapter_smoke:
            command.extend(["--adapter-smoke", "--no-decision-run"])
        if args.overwrite_output:
            command.append("--overwrite-output")
        if args.no_update_best:
            command.append("--no-update-best")
        if args.allow_dirty_forbidden:
            command.append("--allow-dirty-forbidden")
        command.append("--allow-dirty-results")
        returncode, _stdout, _stderr, elapsed = run_command(command)
        summary = load_summary(output_dir)
        run_results.append(
            {
                "run_id": run_id,
                "benchmark": rel_path,
                "required": required,
                "adapter_smoke": adapter_smoke,
                "returncode": returncode,
                "elapsed_seconds": elapsed,
                "summary": summary,
            }
        )
        if returncode:
            failures.append((run_id, returncode, required))
            if args.stop_on_failure or short_name == "synthetic":
                break

    manifest = {
        "schema_version": 1,
        "run_prefix": run_prefix,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_context_path": str(test_context_path),
        "test_context": test_context,
        "runs": run_results,
        "recommendation": "keep" if not any(required for _run_id, _returncode, required in failures) else "revert",
    }
    manifest_path = args.output_root / f"{run_prefix}_matrix_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if failures:
        print("matrix failures:")
        for run_id, returncode, required in failures:
            print(f"- {run_id}: exit={returncode} required={required}")
        return 1 if any(required for _run_id, _returncode, required in failures) else 0
    print(f"matrix complete: run_prefix={run_prefix} runs={len(matrix)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
