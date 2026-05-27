from __future__ import annotations


def judge_mqm_none(*_args, **_kwargs) -> dict[str, object]:
    return {"enabled": False, "critical_errors": 0, "major_errors": 0, "minor_errors": 0, "errors": []}
