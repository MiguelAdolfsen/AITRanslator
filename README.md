# Local Manga Translator

Local-first manga page translator for personal use.

It takes a folder of images, finds Japanese text with OCR, translates it to English with an offline translator, removes the original text, draws the English text back onto the page, and writes translated images to an output folder.

This is a practical MVP, not a scanlation-quality editor. It works best on clear black text inside white speech bubbles. Sound effects, heavy art backgrounds, furigana, and dense vertical text will still need manual cleanup.

## What it uses

- **Text detection:** local Comic Text Detector model, with Tesseract/visual fallbacks.
- **OCR:** local `manga-ocr` crop reader, with Tesseract as a fallback.
- **Translation:** local OPUS Japanese-English model by default, no paid API. MADLAD and Argos remain optional.
- **Context translation:** optional local Qwen GGUF model through Ollama or `llama-cpp-python`.
- **Image editing:** OpenCV inpainting or simple white box erase.
- **Typesetting:** Pillow text fitting.

## Setup

1. Install Python 3.10+.
2. Install Tesseract OCR:
   - Windows: install from the UB Mannheim build: https://github.com/UB-Mannheim/tesseract/wiki
   - During install, include Japanese language data if available.
   - If Japanese data is not included, install `jpn.traineddata` and `jpn_vert.traineddata`.
   - The app auto-detects `C:\Program Files\Tesseract-OCR\tesseract.exe`. If Tesseract is installed elsewhere, select `tesseract.exe` with the GUI's Tesseract path Browse button.
3. Install Python dependencies:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

4. Install the Comic Text Detector model:

```powershell
python -m manga_local_translator.install_ctd
```

5. Install the OPUS translator model:

```powershell
python -m manga_local_translator.install_opus
```

Optional large MADLAD setup:

```powershell
python -m manga_local_translator.install_madlad
```

MADLAD is a large model. It downloads once into the Hugging Face cache and runs locally after that.

Optional legacy Argos setup:

```powershell
python -m manga_local_translator.install_argos ja en
```

## Usage

GUI launcher on Windows:

```powershell
.\Launch-GUI.ps1
```

The launcher creates `.venv`, installs dependencies, and opens a small folder-picker GUI.

Each launch recreates `manga_translator_debug.log` in the project folder. Check that file when OCR, dependency installation, translation, or image writing fails.

CLI:

```powershell
python -m manga_local_translator "C:\path\to\raw-pages" "C:\path\to\translated-pages"
```

Useful options:

```powershell
python -m manga_local_translator raw translated --erase-mode white --debug
python -m manga_local_translator raw translated --translator none
python -m manga_local_translator raw translated --translator madlad
python -m manga_local_translator raw translated --translator qwen
python -m manga_local_translator raw translated --translator argos
python -m manga_local_translator raw translated --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
python -m manga_local_translator raw translated --font "C:\Windows\Fonts\arial.ttf"
python -m manga_local_translator raw translated --glossary translation_glossary.json
```

Output preserves the input folder structure.

For Qwen, put a `.gguf` file under `.models\qwen\`. The app auto-detects the newest `.gguf`; with Ollama installed it creates a local `manga-qwen` model on first use.

## Translation glossary

OPUS is much better when kana-only OCR text is cleaned up before translation. The app includes a small built-in cleanup list for common manga/chapter terms and can also load a custom `translation_glossary.json` from the project folder, or from `--glossary`.

Use `translation_glossary.example.json` as a starting point. Custom entries can:

- replace OCR/source text before translation,
- force exact phrase translations,
- fix recurring English names after translation.

## Notes

- `--erase-mode white` is usually cleaner for normal speech bubbles.
- `--erase-mode inpaint` is better when text sits over artwork, but can smear screentones.
- If OCR finds nothing, confirm Tesseract can see Japanese:

```powershell
tesseract --list-langs
```

You should see `jpn` and ideally `jpn_vert`.
