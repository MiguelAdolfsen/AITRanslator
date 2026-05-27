from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

RESULT_FIELDS = [
    "timestamp",
    "run_id",
    "benchmark_id",
    "reference_policy",
    "experiment_name",
    "translation_quality_score",
    "case_count",
    "candidate_count",
    "hard_reject",
    "hard_failure_rate",
    "critical_mqm_errors",
    "japanese_leakage_count",
    "assistant_chatter_count",
    "glossary_violation_count",
    "notes",
]


def append_result_row(path: str | Path, row: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not p.exists() or p.stat().st_size == 0
    with p.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t", lineterminator="\n")
        if needs_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})
