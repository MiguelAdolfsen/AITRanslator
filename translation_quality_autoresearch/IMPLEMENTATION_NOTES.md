# Implementation Notes

The initial harness is standard-library only. Replay evaluation scores frozen candidates with deterministic hard checks, reference-token overlap, lightweight manga-style proxies, MQM placeholder records, and deterministic reranking.

Live generation imports main-project translators dynamically and records source text, surrounding context, glossary terms, candidates, raw outputs, prompt hashes, and agent warnings. Unavailable agents are recorded as warnings unless `--strict-agents` is passed.

Evaluation writes canonical handoff artifacts (`case_results.jsonl`, `traces.jsonl`, `failures.jsonl`, and `artifacts/run_manifest.json`) plus legacy inspection files (`candidate_scores.jsonl`, `case_decisions.jsonl`, and `failure_table.tsv`). Comparisons reject mismatched benchmark ids, reference policies, benchmark version paths, or case counts.

`benchmark_v1/` and `benchmark_holdout_v1/` are manual Codex handoff benchmarks. Their references were manually translated from decoded OCR source/context, while frozen baseline outputs come from mined quality-run draft translations. They are useful for early agent iteration but should not be treated as native-reviewed human-gold labels.
