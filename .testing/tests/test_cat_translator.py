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
    build_cat_prompt,
    cat_bypass_reason,
    cat_num_predict,
    cat_ollama_model_name,
    cat_reject_reason,
    clean_cat_output,
    ensure_cat_ollama_model,
    find_cat_gguf_path,
    finalize_cat_translation,
)
from manga_local_translator.pipeline import (
    apply_qwen_fallback_translations,
    retry_cat_failures,
    should_try_qwen_fallback,
    translation_cache_stage,
)
from manga_local_translator.page_types import PreparedPage


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
            "manga-cat-cat-translate-7b-q8-0-hfchat",
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

    def test_ensure_cat_ollama_model_uses_hf_chat_template(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            gguf = Path(temp_dir) / "model.gguf"
            gguf.write_text("model", encoding="utf-8")
            with patch("manga_local_translator.qwen_ollama.ollama_model_exists", return_value=False), patch(
                "subprocess.run"
            ) as run:
                run.return_value.returncode = 0
                run.return_value.stdout = ""
                run.return_value.stderr = ""

                ensure_cat_ollama_model("ollama", "manga-cat-test", gguf)

            modelfile = gguf.parent / "manga-cat-test.Modelfile"
            text = modelfile.read_text(encoding="utf-8")
            self.assertIn("<|im_start|>system", text)
            self.assertIn("<|im_start|>user", text)
            self.assertIn("<|im_start|>assistant", text)
            self.assertIn('SYSTEM """You are a helpful assistant."""', text)

    def test_clean_cat_output_strips_harmless_prefix(self) -> None:
        self.assertEqual(clean_cat_output('Translation: "Hello."'), "Hello.")
        self.assertEqual(clean_cat_output('English: "Hello."'), "Hello.")

    def test_cat_prompts_use_short_default_and_source_only_retry(self) -> None:
        self.assertEqual(build_cat_prompt("\u6bcd"), "Translate the following Japanese text into English.\n\n\u6bcd")
        self.assertEqual(build_cat_prompt("\u6bcd", retry=True), "Japanese:\n\u6bcd\n\nEnglish:")

    def test_cat_num_predict_reads_bounded_environment_value(self) -> None:
        with patch.dict("os.environ", {"MANGA_CAT_NUM_PREDICT": "48"}):
            self.assertEqual(cat_num_predict(), 48)
        with patch.dict("os.environ", {"MANGA_CAT_NUM_PREDICT": "2"}):
            self.assertEqual(cat_num_predict(), 16)
        with patch.dict("os.environ", {"MANGA_CAT_NUM_PREDICT": "bad"}):
            self.assertEqual(cat_num_predict(), 128)

    def test_cat_bypass_reason_catches_short_and_noisy_sources(self) -> None:
        self.assertIsNone(cat_bypass_reason("\u4f55\u3042\u308c"))
        with patch.dict("os.environ", {"MANGA_CAT_BYPASS_RISKY_SOURCE": "1"}):
            self.assertEqual(cat_bypass_reason("\u4f55\u3042\u308c"), "short_ambiguous_fragment")
            self.assertEqual(cat_bypass_reason("\u25bd\u65e5\uff11\u56de\u5065\u4eba\u300e\u30a2\u30d9\u30c4\u30aa\u30ca"), "noisy_credit_or_metadata")
            self.assertEqual(cat_bypass_reason("\u30a6\u30a3\u30ea\u30a2\u30e0"), "short_ambiguous_fragment")
            self.assertIsNone(cat_bypass_reason("\u304a\u307e\u3048\u306f\u304a\u98a8\u5442\u306b\u4ed8\u304f\u5408\u3044\u306f\u305a\u3060\u3063\u305f\u3060\u3051\u3058\u3083\u306a\u3044\u3002"))

    def test_cat_reject_reason_catches_chatter_prompt_and_japanese(self) -> None:
        self.assertEqual(
            cat_reject_reason("母", "Could you please provide the Japanese passage?"),
            "cat_chatter",
        )
        self.assertEqual(
            cat_reject_reason("\u6bcd", "Translated by: The user provided a Japanese sentence."),
            "cat_chatter",
        )
        self.assertEqual(
            cat_reject_reason("\u6bcd", "Assist with the translation of Japanese to English."),
            "cat_chatter",
        )
        self.assertEqual(
            cat_reject_reason("\u6bcd", "I'm sorry, but I can't help with that. Feel free to ask!"),
            "cat_chatter",
        )
        self.assertEqual(
            cat_reject_reason("\u6bcd", "I don't understand the Japanese text. Could you please clarify?"),
            "cat_chatter",
        )
        self.assertEqual(
            cat_reject_reason("母", "Translate the following Japanese text into English."),
            "cat_prompt_fragment",
        )
        self.assertEqual(cat_reject_reason("母", "母です"), "cat_untranslated_japanese")
        self.assertEqual(
            cat_reject_reason("\u6bcd", "Japanese: English:"),
            "cat_prompt_fragment",
        )
        self.assertEqual(
            cat_reject_reason("\u6bcd", "William " * 12),
            "cat_verbose_or_repetitive",
        )

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

    def test_cat_retry_runs_only_after_primary_rejection_and_updates_translation(self) -> None:
        class FakeCat:
            def __init__(self) -> None:
                self.debug = {}

            def retry_translation(self, text):
                self.debug[text] = {
                    "cat_retry_attempted": True,
                    "cat_retry_accepted": True,
                    "cat_retry_raw_translation": "English: Mother.",
                    "cat_retry_cleaned_translation": "Mother.",
                    "cat_retry_final": "Mother.",
                    "cat_retry_rejected": False,
                    "cat_retry_reject_reason": "",
                    "cat_rejected": False,
                    "cat_reject_reason": "",
                    "cat_final": "Mother.",
                }
                return "Mother."

            def debug_info_for(self, text, debug_id=None):
                return self.debug.get(text, {})

        block = TextBlock("\u6bcd", (0, 0, 10, 10), 90)
        block.metadata["line_id"] = "line-1"
        page = PreparedPage(
            image_path=Path("1.png"),
            output_path=Path("out.png"),
            image_bgr=None,
            width=10,
            height=10,
            raw_blocks=[block],
            render_blocks=[block],
            skipped_blocks=[],
            grouping_report=[],
            page_order_report=[{"line_id": "line-1", "source_text": "\u6bcd", "page_order": 1}],
            translations={"line-1": ""},
            translation_contexts={
                "line-1": {
                    "primary_translator": "cat",
                    "cat_rejected": True,
                    "cat_reject_reason": "cat_chatter",
                }
            },
        )

        attempted, accepted = retry_cat_failures(page, FakeCat(), PipelineConfig(translator="cat"))

        self.assertEqual((attempted, accepted), (1, 1))
        self.assertEqual(page.translations["line-1"], "Mother.")
        self.assertTrue(page.translation_contexts["line-1"]["cat_retry_attempted"])
        self.assertTrue(page.translation_contexts["line-1"]["cat_retry_accepted"])
        self.assertFalse(page.translation_contexts["line-1"]["cat_rejected"])

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
