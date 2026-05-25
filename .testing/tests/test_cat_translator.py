from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from manga_local_translator.config import PipelineConfig
from manga_local_translator.detect_types import TextBlock
from manga_local_translator import hf_translators
from manga_local_translator.hf_translators import (
    CAT_PROMPT_VERSION,
    CAT_VALIDATION_VERSION,
    cat_ollama_model_name,
    cat_reject_reason,
    clean_cat_output,
    ensure_cat_ollama_model,
    find_cat_gguf_path,
    finalize_cat_translation,
)
from manga_local_translator.pipeline import (
    apply_qwen_fallback_translations,
    should_try_qwen_fallback,
    translation_cache_stage,
)


class CatTranslatorTests(unittest.TestCase):
    def test_find_cat_gguf_prefers_q8_local_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            q4 = model_dir / "CAT-Translate-7b.Q4_K_M.gguf"
            q8 = model_dir / "CAT-Translate-7b.Q8_0.gguf"
            q4.write_text("q4", encoding="utf-8")
            q8.write_text("q8", encoding="utf-8")

            with patch.object(hf_translators, "CAT_MODEL_DIR", model_dir):
                self.assertEqual(find_cat_gguf_path(), q8.resolve())

    def test_find_cat_gguf_respects_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            explicit = Path(temp_dir) / "custom.gguf"
            explicit.write_text("model", encoding="utf-8")

            self.assertEqual(find_cat_gguf_path(explicit), explicit.resolve())

    def test_cat_ollama_model_name_is_stable(self) -> None:
        self.assertEqual(
            cat_ollama_model_name(Path("CAT-Translate-7b.Q8_0.gguf")),
            "manga-cat-cat-translate-7b-q8-0",
        )

    def test_ensure_cat_ollama_model_reuses_existing_model(self) -> None:
        with patch("manga_local_translator.qwen_ollama.ollama_model_exists", return_value=True):
            self.assertEqual(
                ensure_cat_ollama_model("ollama", "manga-cat-existing", Path("model.gguf")),
                "manga-cat-existing",
            )

    def test_ensure_cat_ollama_model_reports_create_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            gguf = Path(temp_dir) / "model.gguf"
            gguf.write_text("model", encoding="utf-8")
            with patch("manga_local_translator.qwen_ollama.ollama_model_exists", return_value=False), patch(
                "subprocess.run"
            ) as run:
                run.return_value.returncode = 1
                run.return_value.stdout = ""
                run.return_value.stderr = "boom"

                with self.assertRaisesRegex(RuntimeError, "Could not create Ollama model"):
                    ensure_cat_ollama_model("ollama", "manga-cat-test", gguf)

    def test_clean_cat_output_strips_harmless_prefix(self) -> None:
        self.assertEqual(clean_cat_output('Translation: "Hello."'), "Hello.")

    def test_cat_reject_reason_catches_chatter_prompt_and_japanese(self) -> None:
        self.assertEqual(
            cat_reject_reason("母", "Could you please provide the Japanese passage?"),
            "cat_chatter",
        )
        self.assertEqual(
            cat_reject_reason("母", "Translate the following Japanese text into English."),
            "cat_prompt_fragment",
        )
        self.assertEqual(cat_reject_reason("母", "母です"), "cat_untranslated_japanese")

    def test_finalize_cat_translation_rejects_bad_output(self) -> None:
        cleaned, final, reason = finalize_cat_translation(
            "母",
            "母",
            "I don't see any Japanese text here.",
            None,
        )

        self.assertEqual(cleaned, "")
        self.assertEqual(final, "")
        self.assertEqual(reason, "cat_chatter")

    def test_cat_rejected_translation_triggers_qwen_fallback(self) -> None:
        self.assertTrue(
            should_try_qwen_fallback(
                "母",
                "",
                {"primary_translator": "cat", "cat_used": True, "cat_rejected": True},
            )
        )

    def test_qwen_fallback_marks_cat_q8_rescue(self) -> None:
        class FakeFallback:
            def __init__(self) -> None:
                self.debug = {}

            def translate_with_context(self, text, **_kwargs):
                self.debug[text] = {"qwen_model": "fake-q8"}
                return "Mom."

            def debug_info_for(self, text):
                return self.debug.get(text, {})

        block = TextBlock("母", (0, 0, 10, 10), 90)
        translations = {"line-1": ""}
        contexts = {
            "line-1": {
                "primary_translator": "cat",
                "cat_used": True,
                "cat_rejected": True,
                "cat_reject_reason": "cat_empty",
            }
        }
        block.metadata["line_id"] = "line-1"

        attempted, accepted = apply_qwen_fallback_translations(
            [block],
            translations,
            contexts,
            [{"line_id": "line-1", "source_text": block.text, "box": block.box, "page_order": 1}],
            FakeFallback(),
            PipelineConfig(translator="cat"),
        )

        self.assertEqual((attempted, accepted), (1, 1))
        self.assertTrue(contexts["line-1"]["cat_q8_fallback_attempted"])
        self.assertTrue(contexts["line-1"]["cat_q8_fallback_accepted"])

    def test_cat_cache_stage_includes_model_prompt_and_validation_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Path(temp_dir) / "CAT-Translate-7b.Q8_0.gguf"
            other = Path(temp_dir) / "CAT-Translate-7b.Q4_K_M.gguf"
            model.write_text("q8", encoding="utf-8")
            other.write_text("q4", encoding="utf-8")

            first = translation_cache_stage(PipelineConfig(translator="cat", cat_model_name=str(model)), "primary")
            second = translation_cache_stage(PipelineConfig(translator="cat", cat_model_name=str(other)), "primary")

        self.assertIn(CAT_PROMPT_VERSION, first)
        self.assertIn(CAT_VALIDATION_VERSION, first)
        self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
