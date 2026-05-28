# Configuration

## Start Here

`PipelineConfig` in `manga_local_translator/config.py` is the shared runtime contract. CLI and GUI should populate the same fields when they expose the same behavior.

## PipelineConfig Fields

| Field | Default | Meaning |
|---|---:|---|
| `detector` | `ctd` | Text region detector: `ctd`, `tesseract`, or `visual`. |
| `ocr_engine` | `manga-ocr` | OCR backend: `manga-ocr` or `tesseract`. |
| `translator` | `opus` | Translation backend: `opus`, `cat`, `hy-mt2`, `qwen`, `madlad`, `argos`, or `none`. |
| `qwen_mode` | `block` | Qwen strategy: `block` or `page`. |
| `erase_mode` | `white` | Original text removal mode: `white` or `inpaint`. |
| `tesseract_cmd` | `None` | Optional path to `tesseract.exe`. |
| `tesseract_lang` | `jpn+jpn_vert` | Tesseract language string. |
| `tesseract_psm` | `11` | Tesseract page segmentation mode. |
| `min_confidence` | `20.0` | OCR confidence threshold. |
| `padding` | `8` | Pixels around detected text for erasing. |
| `render_expand` | `2.2` | Expansion factor for render boxes. |
| `font_path` | `None` | Optional TTF/OTF font path. |
| `glossary_path` | `None` | Optional glossary JSON path. |
| `qwen_model_path` | `None` | Primary Qwen text model path. |
| `qwen_fallback_model_path` | `None` | Qwen fallback model for rejected/suspicious blocks. |
| `qwen_critic_model_path` | `None` | Qwen critic/verifier model. |
| `cat_model_name` | `None` | CAT GGUF path or Hugging Face model id. |
| `hy_mt2_model_name` | `None` | Tencent Hy-MT2 Hugging Face model id or local model folder. |
| `vision_enabled` | `False` | Enable Qwen vision repair. |
| `vision_facts_enabled` | `False` | Enable pre-translation visual facts. |
| `vision_mode` | `numbered_page` | Vision artifact mode: `numbered_page` or `page_image`. |
| `vision_trigger` | `suspicious` | Vision repair trigger: `suspicious`, `layout`, or `all`. |
| `vision_model_path` | `None` | Optional vision-capable Qwen GGUF. |
| `vision_projector_path` | `None` | Optional multimodal projector GGUF. |
| `base_font_size` | `28` | Starting render font size. |
| `recursive` | `True` | Process nested folders. |
| `overwrite` | `False` | Replace existing outputs. |
| `debug` | `False` | Write debug images/reports. |
| `resume` | `False` | Reuse work cache. |
| `work_dir` | `None` | Optional cache directory. |
| `skip_render` | `False` | Write debug/report data without final images. |
| `render_only` | `False` | Render from existing cache only. |

## CLI Options

`manga_local_translator/cli.py` exposes all `PipelineConfig` fields needed for batch processing. Confirm current behavior with:

```powershell
python -m manga_local_translator --help
```

Current long options:

| Category | Flags |
|---|---|
| Detection/OCR | `--detector`, `--ocr-engine`, `--tesseract-cmd`, `--tesseract-lang`, `--psm`, `--min-confidence` |
| Translation | `--translator`, `--glossary`, `--cat-model`, `--hy-mt2-model`, `--qwen-model`, `--qwen-fallback-model`, `--qwen-critic-model`, `--qwen-mode` |
| Vision | `--vision`, `--vision-facts`, `--vision-mode`, `--vision-trigger`, `--vision-model`, `--vision-projector` |
| Rendering | `--erase-mode`, `--padding`, `--render-expand`, `--font`, `--font-size`, `--skip-render`, `--render-only` |
| Run control | `--recursive`, `--no-recursive`, `--overwrite`, `--debug`, `--resume`, `--work-dir` |

When adding a CLI option:

1. Add parser argument in `cli.py`.
2. Add or update the field in `PipelineConfig`.
3. Pass the parsed value into `PipelineConfig`.
4. Add GUI support in `gui.py` if normal users should access it.
5. Update README and this page.

## Model Paths

| Backend | Default/search behavior | Override |
|---|---|---|
| CTD | installed assets managed by `ctd_setup.py` | `python -m manga_local_translator.install_ctd` |
| OPUS | Hugging Face cache | `python -m manga_local_translator.install_opus` |
| Tencent Hy-MT2 | Hugging Face cache, default `tencent/Hy-MT2-1.8B` | `python -m manga_local_translator.install_hy_mt2`, `--hy-mt2-model` |
| MADLAD | Hugging Face cache | `python -m manga_local_translator.install_madlad` |
| Argos | Argos package data | `python -m manga_local_translator.install_argos ja en` |
| CAT | `.models/CAT-Translate/*.gguf`, then default model id | `--cat-model` |
| Qwen text | `.models/qwen` preferred text model | `--qwen-model`, `MANGA_QWEN_MODEL` |
| Qwen vision | selected vision model plus projector | `--vision-model`, `--vision-projector` |

Do not assume `.models/` exists in a fresh checkout.

## Environment Variables

The Qwen code supports generation overrides with prefixes such as:

```text
MANGA_QWEN_TRANSLATION_*
MANGA_QWEN_PAGE_*
MANGA_QWEN_VERIFICATION_*
MANGA_QWEN_REPAIR_*
```

`quality_eval.py` sets profile-specific Qwen env vars for repeatable experiments.

## Glossary

Glossary behavior lives in `translation_rules.py`. The example file is `translation_glossary.example.json`.

Glossaries can:

- normalize source text before translation,
- force exact phrase translations,
- replace target text after translation.

If glossary schema changes, update README, this page, and relevant translation rule tests.
