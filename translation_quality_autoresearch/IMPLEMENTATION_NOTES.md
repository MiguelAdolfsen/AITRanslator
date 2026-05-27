# Implementation Notes

The initial harness is standard-library only. Replay evaluation scores frozen candidates with deterministic hard checks, reference-token overlap, lightweight manga-style proxies, MQM placeholder records, and deterministic reranking.

Live generation imports main-project translators dynamically and records traces. Unavailable agents are recorded as warnings unless `--strict-agents` is passed.

`benchmark_v1/` and `benchmark_holdout_v1/` are manual Codex handoff benchmarks. Their references were manually translated from decoded OCR source/context, while frozen baseline outputs come from mined quality-run draft translations. They are useful for early agent iteration but should not be treated as native-reviewed human-gold labels.
