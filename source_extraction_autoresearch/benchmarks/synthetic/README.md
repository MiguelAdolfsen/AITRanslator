# Synthetic Source Extraction Benchmark

Deterministic generated pages for harness smoke testing and regression checks.
These fixtures are not a substitute for real hand-labeled manga pages.

Coverage includes vertical and horizontal dialogue, narration boxes, signs, thought bubbles, split text groups, close but separate bubbles, low-contrast text, page-edge regions, large and nearby SFX, mixed Japanese/Latin signs, small furigana-like text, honorific fragments, overlapping bubble risk, reading-order stress, dense pages, and ignored metadata/credit/page-number/noise regions.

The benchmark is intentionally deterministic. Add new generic manga/source extraction failure patterns here before tuning the extraction code, but avoid case-specific rules that only match these generated labels.

Font: C:\Windows\Fonts\msgothic.ttc
