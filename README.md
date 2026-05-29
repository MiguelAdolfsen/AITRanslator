# AI Translator / Local Manga Translator

A local-first Japanese manga image translator.

The project takes manga page images, detects Japanese text regions, OCRs the text, translates it to English, removes or covers the original text, and renders the translated text back into the image.

It is designed for local experimentation and personal translation workflows. It includes both a GUI and a CLI, plus optional Qwen-based translation, verification, visual-facts enrichment, and vision-assisted repair.

Agents should start with [`CODEX.md`](CODEX.md), then use the deeper [`docs/agent-wiki/README.md`](docs/agent-wiki/README.md) project wiki. Skill, issue-tracker, label, and domain conventions live under [`docs/agents/`](docs/agents/).

---

## What the pipeline does

For each input image, the translator can:

1. Load one image or recursively scan a folder of images.
2. Detect manga text regions.
3. OCR Japanese text from each detected region.
4. Filter bad OCR fragments and non-Japanese noise.
5. Group text regions into render blocks.
6. Translate text with the selected translator.
7. Optionally review and repair translations with CAT retry, Qwen critic checks, Qwen fallback, or cached translation stages.
8. Optionally use Qwen visual facts and numbered page artifacts for page-aware repair.
9. Erase or cover the original text.
10. Typeset the English translation back into the page.
11. Write the translated image, cache artifacts, and optional debug reports.

Supported image extensions include:

```text
.jpg
.jpeg
.png
.webp
.bmp
.tif
.tiff
```

---

## Main features

- Tkinter GUI launcher for normal use.
- CLI for batch processing and experiments.
- Recursive folder processing.
- Text detection with:
  - Comic Text Detector / CTD
  - Tesseract
  - lightweight visual detector
- OCR with:
  - manga-ocr
  - Tesseract
- Translation with:
  - OPUS-MT Japanese to English
  - Tencent Hy-MT2
  - MADLAD-400
  - Argos Translate
  - local Qwen GGUF models
  - no-translation debug mode
- Qwen translation modes:
  - block mode
  - page mode
- Optional Qwen verifier and repair flow.
- Optional Qwen vision repair using numbered page artifacts.
- Optional Qwen visual-facts pass.
- Glossary support.
- White-box or inpainting erase modes.
- Pillow-based text rendering.
- Resume/cache support.
- OCR/debug JSON reports.
- Debug images with boxes and render diagnostics.
- Repeatable quality-evaluation runner.
- Render-only benchmark reruns from cache.
- Isolated render/layout autoresearch benchmark harness.

---

## Current limitations

This is not a fully automatic professional lettering system.

Expect manual review for:

- stylized sound effects
- complex background text
- dense vertical layouts
- tiny bubbles
- furigana
- unusual fonts
- handwritten text
- low-resolution scans
- heavily textured or colored bubbles
- ambiguous speakers
- OCR mistakes
- pages where the original art needs careful redrawing

The project works best when text is clear, dark, and inside white or lightly colored speech bubbles.

---

## Requirements

### Required

- Python 3.10 or newer
- A working Python virtual environment
- Internet access for first-time model downloads
- Enough disk space for OCR / translation models

### Strongly recommended

- Windows, especially if using the included PowerShell GUI launcher
- Tesseract OCR with Japanese language data
- `jpn` and `jpn_vert` Tesseract languages
- A GPU for faster local model inference, if available

### Optional

- Ollama for Qwen GGUF translation
- `llama-cpp-python` for local GGUF inference without Ollama
- Qwen GGUF model files
- Qwen multimodal projector file for vision features
- Tencent Hy-MT2 Hugging Face cache files if using Hy-MT2
- Argos Translate package if using Argos

---

## Quick start: Windows GUI

Clone the repository:

```powershell
git clone https://github.com/MiguelAdolfsen/AITRanslator.git
cd AITRanslator
```

Launch the GUI:

```powershell
.\Launch-GUI.ps1
```

The launcher will:

1. Create `.venv` if it does not already exist.
2. Install packages from `requirements.txt`.
3. Set useful cache/runtime environment variables.
4. Start the GUI with:

```powershell
python -m manga_local_translator.gui
```

In the GUI:

1. Select an input image or folder.
2. Select an output file or folder.
3. Choose detector, OCR engine, translator, and erase mode.
4. Install CTD / OPUS / MADLAD / Argos from the GUI if needed.
5. Enable debug output while tuning settings.
6. Click **Start**.

---

## Manual setup

Create and activate a virtual environment:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Upgrade pip and install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install the default detector and translator models:

```powershell
python -m manga_local_translator.install_ctd
python -m manga_local_translator.install_opus
```

You can then run the CLI:

```powershell
python -m manga_local_translator --help
```

If the package is installed with its entry point available, this also works:

```powershell
manga-local-translator --help
```

---

## Tesseract setup

Tesseract is optional if you use CTD plus manga-ocr, but it is useful as a fallback detector/OCR path.

Install Tesseract and verify the Japanese language files:

```powershell
tesseract --list-langs
```

Recommended languages:

```text
jpn
jpn_vert
```

If Tesseract is not on your PATH, pass the executable path manually:

```powershell
python -m manga_local_translator input output `
  --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

The GUI also has a Tesseract path field.

---

## Model setup

### CTD detector

Install Comic Text Detector assets:

```powershell
python -m manga_local_translator.install_ctd
```

Use it with:

```powershell
--detector ctd
```

This is the default detector.

---

### OPUS translator

Install the default OPUS Japanese-to-English model:

```powershell
python -m manga_local_translator.install_opus
```

Use it with:

```powershell
--translator opus
```

This is the default translator.

---

### CAT translator

CAT is an optional local quality-mode translator. It is intended for experiments as a primary pass before Qwen critic/Q8 repair.

Default local GGUF search location:

```text
.models/CAT-Translate
```

Preferred filename:

```text
CAT-Translate-7b.Q8_0.gguf
```

Use it with:

```powershell
python -m manga_local_translator input output `
  --translator cat `
  --cat-model ".models\CAT-Translate\CAT-Translate-7b.Q8_0.gguf"
```

Recommended benchmark setup:

```powershell
python -m manga_local_translator.quality_eval ".testing\input\diverse-smoke-20260524" `
  --output-root quality-runs `
  --name cat-primary-test `
  --profiles quality `
  --translator cat `
  --cat-model ".models\CAT-Translate\CAT-Translate-7b.Q8_0.gguf" `
  --qwen-critic-model ".models\qwen\Qwen3-8B-Q4_K_M.gguf" `
  --qwen-fallback-model ".models\qwen\Qwen3.5-9B-Q8_0.gguf" `
  --resume
```

CAT-only benchmark with image rendering enabled:

```powershell
python -m manga_local_translator.quality_eval ".testing\input\diverse-smoke-20260524" `
  --output-root quality-runs `
  --name cat-only-render-test `
  --profiles quality `
  --translator cat `
  --cat-model ".models\CAT-Translate\CAT-Translate-7b.Q8_0.gguf" `
  --render-images
```

CAT runs locally through Ollama for GGUF files. Runtime does not download models; the install script only downloads Hugging Face cache files when you explicitly run it:

```powershell
python -m manga_local_translator.install_cat
```

---

### Tencent Hy-MT2 translator

Install the default Tencent Hy-MT2 model:

```powershell
python -m manga_local_translator.install_hy_mt2
```

Use it with:

```powershell
python -m manga_local_translator input output `
  --translator hy-mt2
```

The default model is `tencent/Hy-MT2-1.8B`. You can point at another cached Hy-MT2 model id or local model folder:

```powershell
python -m manga_local_translator input output `
  --translator hy-mt2 `
  --hy-mt2-model "tencent/Hy-MT2-7B"
```

Hy-MT2 runs through Hugging Face Transformers using local cache files at runtime. It does not download during translation; run the install command first.

---

### MADLAD translator

Install MADLAD:

```powershell
python -m manga_local_translator.install_madlad
```

Use it with:

```powershell
--translator madlad
```

MADLAD is larger than OPUS and may be slower.

---

### Argos translator

Install Argos Translate support:

```powershell
pip install argostranslate
python -m manga_local_translator.install_argos ja en
```

Use it with:

```powershell
--translator argos
```

---

### Qwen translator

Qwen support is for local GGUF models.

The code looks for Qwen models in:

```text
.models/qwen
```

You can also pass a model directly:

```powershell
python -m manga_local_translator input output `
  --translator qwen `
  --qwen-model ".models\qwen\your-qwen-model.gguf"
```

You can provide a fallback model:

```powershell
python -m manga_local_translator input output `
  --translator qwen `
  --qwen-model ".models\qwen\primary.gguf" `
  --qwen-fallback-model ".models\qwen\fallback.gguf"
```

Qwen can use:

- Ollama, when available
- `llama-cpp-python`, as a local fallback path

You can also set:

```powershell
$env:MANGA_QWEN_MODEL = ".models\qwen\your-qwen-model.gguf"
```

---

### Qwen vision

Vision support is optional and advanced.

Vision features require:

- a vision-capable Qwen GGUF model
- a multimodal projector / mmproj GGUF file
- `llama-cpp-python` with Qwen vision chat-handler support

Default projector search location:

```text
.models/qwen
```

Default projector-style filename expected by the code:

```text
mmproj-Qwen3.5-9B-BF16.gguf
```

You can pass model and projector paths manually:

```powershell
python -m manga_local_translator input output `
  --translator qwen `
  --vision `
  --vision-model ".models\qwen\vision-model.gguf" `
  --vision-projector ".models\qwen\mmproj-model.gguf"
```

Vision is used as context and repair support. It should not replace OCR.

---

## CLI usage

Basic command:

```powershell
python -m manga_local_translator INPUT OUTPUT [options]
```

Input and output may be files or folders.

Example: translate a folder recursively with defaults:

```powershell
python -m manga_local_translator ".\raw_pages" ".\translated_pages"
```

Example: enable debug output and overwrite existing files:

```powershell
python -m manga_local_translator ".\raw_pages" ".\translated_pages" `
  --debug `
  --overwrite
```

Example: use Tesseract detector and OCR:

```powershell
python -m manga_local_translator ".\raw_pages" ".\translated_pages" `
  --detector tesseract `
  --ocr-engine tesseract `
  --tesseract-lang jpn+jpn_vert `
  --psm 11
```

Example: use Qwen in block mode:

```powershell
python -m manga_local_translator ".\raw_pages" ".\translated_pages" `
  --translator qwen `
  --qwen-mode block `
  --qwen-model ".models\qwen\qwen-model.gguf"
```

Example: use Qwen page mode with resume support:

```powershell
python -m manga_local_translator ".\raw_pages" ".\translated_pages" `
  --translator qwen `
  --qwen-mode page `
  --resume `
  --work-dir ".\.manga-work"
```

Example: use Qwen with vision repair:

```powershell
python -m manga_local_translator ".\raw_pages" ".\translated_pages" `
  --translator qwen `
  --vision `
  --vision-trigger suspicious `
  --vision-mode numbered_page
```

Example: use Qwen visual facts before translation:

```powershell
python -m manga_local_translator ".\raw_pages" ".\translated_pages" `
  --translator qwen `
  --qwen-mode page `
  --vision-facts `
  --vision-mode numbered_page
```

Example: OCR/render debug without translating:

```powershell
python -m manga_local_translator ".\raw_pages" ".\translated_pages" `
  --translator none `
  --debug `
  --overwrite
```

Example: translation benchmark without rendering images:

```powershell
python -m manga_local_translator.quality_eval ".\raw_pages" `
  --output-root ".\quality-runs" `
  --name "translation-only-test" `
  --profiles quality `
  --translator cat `
  --cat-model ".models\CAT-Translate\CAT-Translate-7b.Q8_0.gguf"
```

Example: render images later from an existing benchmark cache without rerunning OCR or translation:

```powershell
python -m manga_local_translator.quality_eval ".\raw_pages" `
  --output-root ".\quality-runs" `
  --name "translation-only-test" `
  --profiles quality `
  --translator cat `
  --cat-model ".models\CAT-Translate\CAT-Translate-7b.Q8_0.gguf" `
  --render-only `
  --resume
```

`--render-only` fails if the matching `.manga-work` prepared/translation cache is missing. Use the same translator/model/critic/fallback flags that created the cached benchmark.

---

## Important CLI options

### Input/output

| Option | Description |
|---|---|
| `input` | Input image or folder |
| `output` | Output image or folder |
| `--recursive` | Recursively process folders |
| `--no-recursive` | Disable recursive folder processing |
| `--overwrite` | Replace existing output files |
| `--resume` | Reuse cached prepared pages/translations when possible |
| `--work-dir PATH` | Override the cache/work directory |
| `--skip-render` | Write debug JSON/reports without final image rendering |
| `--render-only` | Render images from existing cache only; no OCR/translation work |

### Detection and OCR

| Option | Values | Default |
|---|---|---|
| `--detector` | `ctd`, `tesseract`, `visual` | `ctd` |
| `--ocr-engine` | `manga-ocr`, `tesseract` | `manga-ocr` |
| `--tesseract-cmd` | path | unset |
| `--tesseract-lang` | language string | `jpn+jpn_vert` |
| `--psm` | Tesseract page segmentation mode | `11` |
| `--min-confidence` | OCR confidence threshold | `20.0` |

### Translation

| Option | Values | Default |
|---|---|---|
| `--translator` | `opus`, `cat`, `hy-mt2`, `qwen`, `madlad`, `argos`, `none` | `opus` |
| `--glossary` | JSON glossary path | unset |
| `--cat-model` | CAT GGUF path or Hugging Face model id | unset |
| `--hy-mt2-model` | Hy-MT2 Hugging Face model id or local model folder | unset |

### Qwen

| Option | Values | Default |
|---|---|---|
| `--qwen-model` | GGUF path or model name | unset |
| `--qwen-critic-model` | GGUF path or model name | unset |
| `--qwen-fallback-model` | GGUF path or model name | unset |
| `--qwen-mode` | `block`, `page` | `block` |

### Vision

| Option | Values | Default |
|---|---|---|
| `--vision` | enable vision repair | disabled |
| `--vision-facts` | enable visual-facts pass | disabled |
| `--vision-mode` | `numbered_page`, `page_image` | `numbered_page` |
| `--vision-trigger` | `suspicious`, `layout`, `all` | `suspicious` |
| `--vision-model` | vision-capable GGUF path | unset |
| `--vision-projector` | mmproj/projector path | unset |

### Rendering

| Option | Description | Default |
|---|---|---|
| `--erase-mode` | `white` or `inpaint` | `white` |
| `--padding` | pixels around text regions | `8` |
| `--render-expand` | render box expansion factor | `2.2` |
| `--font` | path to font file | unset |
| `--font-size` | base render font size | `28` |

### Debugging

| Option | Description |
|---|---|
| `--debug` | Write debug images and OCR/debug JSON reports |

---

## Glossary support

A glossary can enforce names, terms, and post-processing replacements.

Start from the example file:

```powershell
copy translation_glossary.example.json translation_glossary.json
```

Use it with:

```powershell
python -m manga_local_translator input output `
  --glossary ".\translation_glossary.json"
```

Example structure:

```json
{
  "source_replacements": [
    {
      "source": "あじと",
      "target": "アジト"
    }
  ],
  "exact_phrases": {
    "ただいま": "I'm home."
  },
  "target_replacements": [
    {
      "source": "Ajit",
      "target": "hideout",
      "when_source_contains": "アジト"
    }
  ]
}
```

Glossary concepts:

| Field | Purpose |
|---|---|
| `source_replacements` | Normalize or repair Japanese source text before translation |
| `exact_phrases` | Force a specific English translation for an exact Japanese phrase |
| `target_replacements` | Replace unwanted English output after translation |

---

## Debug output

When `--debug` is enabled, the project writes extra diagnostics.

Common debug files include:

| File | Purpose |
|---|---|
| `manga_translator_debug.log` | Main debug log |
| `*.debug.png` | Page image with detected boxes/layout diagnostics |
| `*.ocr.json` | OCR blocks, skipped blocks, translations, render warnings, and metadata |
| `*.vision.png` | Numbered or page-image artifact sent to the vision model |
| `.manga-work/` | Cached prepared pages and translation stages for resume and render-only runs |

Debug reports are useful for checking:

- detected boxes
- OCR text
- skipped OCR fragments
- reading order
- grouped render blocks
- translation output
- render fitting warnings
- Qwen page summaries
- vision repair/facts summaries

---

## Architecture notes

The pipeline is split into small passes so resume, review, vision repair, and render-only benchmark runs can reuse the same page state:

- `pipeline.py` orchestrates input discovery, page loading, pass sequencing, and final reporting.
- `cache_policy.py` defines prepared-page and translation-stage cache plans, including resume and render-only lookup order.
- `page_cache.py` serializes prepared pages and translated cache stages in `.manga-work/`.
- `translation_review_pass.py` coordinates primary translation, cache hits, critic review, CAT retry, and fallback behavior.
- `translated_state.py` keeps translated line state centralized while review and vision passes update blocks.
- `rendered_page_pass.py` handles erase, layout fitting, optional vision repair, debug metadata, and output image writing.

---

## Quality evaluation

The quality evaluation runner can process pages with repeatable profile settings and summarize debug OCR reports.

Show help:

```powershell
python -m manga_local_translator.quality_eval --help
```

Example run:

```powershell
python -m manga_local_translator.quality_eval ".\raw_pages" `
  --output-root ".\quality-runs" `
  --name "my-qwen-test" `
  --profiles quality,strict `
  --qwen-mode page
```

Example vision comparison:

```powershell
python -m manga_local_translator.quality_eval ".\raw_pages" `
  --output-root ".\quality-runs" `
  --name "vision-comparison" `
  --profiles quality `
  --vision `
  --vision-facts `
  --compare-vision
```

Summarize an existing run:

```powershell
python -m manga_local_translator.quality_eval ".\raw_pages" `
  --output-root ".\quality-runs" `
  --name "my-qwen-test" `
  --summarize-only
```

By default, `quality_eval` skips final image rendering so translation/model experiments run faster. Add `--render-images` when you want translated page images. Add `--render-only --resume` to render from an existing cached run without redoing OCR or translation.

---

## Render autoresearch

`render_autoresearch/` is an isolated benchmark harness for improving only rendering and layout behavior. It does not run OCR, detection, translation, Qwen, CAT, vision, or the GUI. It uses frozen synthetic fixtures:

```text
render_autoresearch/benchmarks/synthetic/
```

Run render unit tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .testing\tests -p test_render.py
```

Regenerate deterministic synthetic cases:

```powershell
.\.venv\Scripts\python.exe render_autoresearch\scripts\generate_synthetic_cases.py `
  --output render_autoresearch\benchmarks\synthetic
```

Run the synthetic render benchmark:

```powershell
.\.venv\Scripts\python.exe render_autoresearch\scripts\render_eval.py `
  --benchmark render_autoresearch\benchmarks\synthetic `
  --output render_autoresearch\runs\baseline `
  --results render_autoresearch\results\results.tsv `
  --run-id baseline `
  --overwrite-output
```

The benchmark writes per-run outputs under `render_autoresearch/runs/` and appends scores to `render_autoresearch/results/results.tsv`. Lower `render_score` is better. See `render_autoresearch/PROGRAM.md` before making render experiments.

---

## Tests

The repository includes a PowerShell test runner under `.testing`.

Run normal tests:

```powershell
.\.testing\run_tests.ps1
```

Run tests plus the optional smoke test:

```powershell
.\.testing\run_tests.ps1 -WithSmoke
```

The smoke test expects a sample image at:

```text
..\mangafolder\1.png
```

The test runner performs checks such as:

- Python compile check
- CLI help check
- quality evaluator help check
- unit tests under `.testing/tests`
- optional smoke translation

---

## Project layout

Important files and folders:

```text
AITRanslator/
  Launch-GUI.ps1
  README.md
  pyproject.toml
  requirements.txt
  translation_glossary.example.json

  manga_local_translator/
    cli.py
    gui.py
    pipeline.py
    config.py
    cache_policy.py
    page_cache.py
    detect_ocr.py
    ctd_detector.py
    hf_translators.py
    qwen_translator.py
    qwen_vision.py
    qwen_validation.py
    translation_review.py
    translation_review_pass.py
    translated_state.py
    vision_service.py
    vision_artifact.py
    rendered_page_pass.py
    render.py
    erase.py
    grouping.py
    quality_eval.py
    install_ctd.py
    install_opus.py
    install_hy_mt2.py
    install_madlad.py
    install_argos.py
    install_cat.py

  docs/
    agent-wiki/
    agents/

  .testing/
    README.md
    run_tests.ps1
    tests/

  render_autoresearch/
    README.md
    PROGRAM.md
    benchmarks/
    results/
    runs/
    scripts/

  quality-runs/
    previous quality/evaluation run outputs
```

---

## Recommended workflows

### Simple local translation

Use this when you want the easiest path:

```powershell
python -m manga_local_translator ".\raw_pages" ".\translated_pages" `
  --detector ctd `
  --ocr-engine manga-ocr `
  --translator opus `
  --debug
```

### OCR/layout debugging

Use this before tuning translation:

```powershell
python -m manga_local_translator ".\raw_pages" ".\debug_output" `
  --translator none `
  --debug `
  --overwrite
```

Inspect the generated `.debug.png` and `.ocr.json` files.

### Higher-quality local Qwen pass

Use this when Qwen models are installed:

```powershell
python -m manga_local_translator ".\raw_pages" ".\translated_pages" `
  --translator qwen `
  --qwen-mode page `
  --resume `
  --debug
```

### Qwen with vision assistance

Use this for pages where speaker, layout, or visual context matters:

```powershell
python -m manga_local_translator ".\raw_pages" ".\translated_pages" `
  --translator qwen `
  --qwen-mode page `
  --vision-facts `
  --vision `
  --vision-trigger suspicious `
  --vision-mode numbered_page `
  --resume `
  --debug
```

---

## Troubleshooting

### `CTD detector requested but not available`

Install CTD assets:

```powershell
python -m manga_local_translator.install_ctd
```

Or use another detector:

```powershell
--detector tesseract
```

---

### OPUS model is missing

Install OPUS:

```powershell
python -m manga_local_translator.install_opus
```

---

### Hy-MT2 model is missing

Install Hy-MT2:

```powershell
python -m manga_local_translator.install_hy_mt2
```

---

### Tesseract is not found

Pass the executable path:

```powershell
--tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

Also verify Japanese language files:

```powershell
tesseract --list-langs
```

---

### manga-ocr downloads on first use

The first `manga-ocr` run may download model files. After that, the project tries to use local cached files.

---

### Qwen model not found

Place a GGUF model under:

```text
.models/qwen
```

Or pass:

```powershell
--qwen-model ".models\qwen\model.gguf"
```

Or set:

```powershell
$env:MANGA_QWEN_MODEL = ".models\qwen\model.gguf"
```

---

### Vision projector not found

Place the multimodal projector file under:

```text
.models/qwen
```

Or pass:

```powershell
--vision-projector ".models\qwen\mmproj-model.gguf"
```

---

### Output files are skipped

Use:

```powershell
--overwrite
```

---

### Translation is bad but OCR is also bad

Do not tune the translator first.

Run:

```powershell
--translator none --debug
```

Inspect `.ocr.json` and `.debug.png`, then adjust detector/OCR settings.

---

### Text is too large or does not fit

Try:

```powershell
--font-size 24
--render-expand 2.8
```

You can also pass a different font:

```powershell
--font "C:\Windows\Fonts\arial.ttf"
```

---

## Environment variables

The Qwen code supports several environment-variable overrides.

Commonly useful:

```powershell
$env:MANGA_QWEN_MODEL = ".models\qwen\model.gguf"
```

Generation settings can be overridden with prefixes such as:

```text
MANGA_QWEN_TRANSLATION_*
MANGA_QWEN_PAGE_*
MANGA_QWEN_VERIFICATION_*
MANGA_QWEN_REPAIR_*
```

Examples:

```powershell
$env:MANGA_QWEN_TRANSLATION_TEMPERATURE = "0.2"
$env:MANGA_QWEN_PAGE_NUM_PREDICT = "768"
$env:MANGA_QWEN_VERIFICATION_TEMPERATURE = "0.1"
```

Use these when benchmarking prompt/model changes.

---

## Notes on vision usage

The project’s vision path is intended to support translation, not replace OCR.

Recommended policy:

```text
OCR/source text is authoritative.
Vision is used for context, layout, speaker hints, visual facts, and repair.
Vision should not invent dialogue or silently correct source text.
```

The `numbered_page` vision mode creates an artifact that maps line IDs to locations on the page. This is usually safer than asking a vision model to read the original manga text directly.

---

## License

No license file is currently included in the repository. Add a license before distributing or accepting external reuse/contributions.
