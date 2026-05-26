# CAT Response Holdout

This benchmark is a small locked holdout for sanity checks after a profile wins the main benchmark.

Rules:

- Do not optimize directly against these lines.
- Do not add source-specific cleanup or prompt text from this folder.
- Use this only after a profile improves the main benchmark without hard failures.
- Do not run this after every failed experiment; frequent holdout inspection makes it easier to overfit.
- If a holdout failure reveals a generic weakness, create new main-benchmark cases in a separate pass instead of tuning directly to this file.
