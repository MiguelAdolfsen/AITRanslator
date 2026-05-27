from __future__ import annotations


def pairwise_none(*_args, **_kwargs) -> dict[str, object]:
    return {"enabled": False, "winner": None, "reason": "pairwise judge disabled"}
