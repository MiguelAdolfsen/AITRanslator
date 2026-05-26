# Keep Experimenting

The user asked Codex to keep running the source extraction autoresearch loop and not stop until explicitly told to stop.

Stay within the program scope:
- source extraction only
- no translation, rendering, erase, vision, CAT, Qwen, OPUS, MADLAD, or Argos calls
- keep benchmark/scoring/results schema frozen
- run tests, fixture validation, and benchmark for each experiment
- keep only changes that improve `source_extraction_score` without violating `PROGRAM.md` guards
