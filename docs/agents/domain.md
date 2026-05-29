# Domain Docs

How engineering skills should consume this repo's domain documentation when exploring the codebase.

## Layout

This is a single-context repo.

- Read `CONTEXT.md` at the repo root before proposing domain-language changes.
- Read `docs/adr/` before changing architecture in an area that has recorded decisions.
- If `docs/adr/` does not exist, proceed silently.

## Use the Glossary Vocabulary

When output names a domain concept, use the term as defined in `CONTEXT.md`. Do not drift to synonyms the glossary explicitly avoids.

If the concept you need is not in the glossary yet, either reconsider the term or note that the domain language needs a follow-up conversation.

## Flag ADR Conflicts

If output contradicts an existing ADR, surface it explicitly rather than silently overriding the decision.
