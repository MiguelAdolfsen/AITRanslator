# Translator Context

This project translates Japanese manga page images locally. The language below names project concepts that should stay stable across code, tests, and agent notes.

## Language

**Translation review pass**:
A post-primary-translation pass that inspects or repairs translated text before rendering, including CAT retry, evidence checks, Qwen critic, and Qwen fallback behavior.
_Avoid_: validation service, critic pipeline, fallback layer

**Cache stage policy**:
The rules that name translation cache stages and choose render-only cache lookup order from runtime settings, model identity, prompt versions, review passes, and vision-facts settings.
_Avoid_: cache helper, stage string logic
