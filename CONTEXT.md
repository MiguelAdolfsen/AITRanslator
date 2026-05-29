# Translator Context

This project translates Japanese manga page images locally. The language below names project concepts that should stay stable across code, tests, and agent notes.

## Language

**Translation review pass**:
A post-primary-translation workflow that inspects or repairs translated text before rendering, including CAT retry, evidence checks, Qwen critic, Qwen fallback behavior, review cache writes, and review summary counts.
_Avoid_: validation service, critic pipeline, fallback layer

**Translation review page**:
A page-level review view over a prepared manga page that lets the Translation review pass run CAT retry, Qwen critic, and Qwen fallback without passing raw render blocks, translation maps, context maps, and page order reports through each call.
_Avoid_: review tuple, page helper, context bundle

**Cache stage policy**:
The rules that name translation cache stages and choose render-only cache lookup order from runtime settings, model identity, prompt versions, review passes, and vision-facts settings.
_Avoid_: cache helper, stage string logic

**Translation cache plan**:
The concrete cache-stage plan for a pipeline run, including prepared-page cache paths, translation cache paths, resume lookup order, render-only lookup order, and final translation cache stage.
_Avoid_: cache path helper, stage-name bundle
