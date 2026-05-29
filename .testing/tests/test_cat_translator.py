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
    CAT_SECOND_RETRY_PROMPT_VERSION,
    CAT_VALIDATION_VERSION,
    build_cat_prompt,
    cat_bypass_reason,
    cat_num_predict,
    cat_ollama_model_name,
    cat_reject_reason,
    cat_second_retry_enabled,
    clean_cat_output,
    ensure_cat_ollama_model,
    find_cat_gguf_path,
    finalize_cat_translation,
    resolve_cat_gguf_path,
    salvage_cat_translation,
    translate_kana_honorific_name_fragment,
)
from manga_local_translator.translation_rules import apply_honorific_title_replacements
from manga_local_translator.pipeline import translation_cache_stage
from manga_local_translator.translation_review import (
    apply_qwen_fallback_translations,
    retry_cat_failures,
    should_try_qwen_fallback,
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

    def test_resolve_cat_gguf_uses_gguf_only_for_default_or_explicit_gguf(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            q8 = model_dir / "CAT-Translate-7b.Q8_0.gguf"
            hf_dir = model_dir / "hf-model"
            q8.write_text("q8", encoding="utf-8")
            hf_dir.mkdir()

            with patch.object(hf_translators, "CAT_MODEL_DIR", model_dir):
                self.assertEqual(resolve_cat_gguf_path("cyberagent/CAT-Translate-7b"), q8.resolve())
                self.assertEqual(resolve_cat_gguf_path(q8), q8.resolve())
                self.assertIsNone(resolve_cat_gguf_path(hf_dir))

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
        self.assertEqual(
            build_cat_prompt("\u6bcd"),
            'Translate exactly. Return only concise English. Do not explain, apologize, ask for clarification, or continue the scene. If the source is incomplete, translate the fragment as a fragment.\nJapanese: "\u6bcd"\nEnglish:',
        )
        self.assertEqual(build_cat_prompt("\u6bcd", retry=True), "Japanese:\n\u6bcd\n\nEnglish:")
        self.assertEqual(
            build_cat_prompt("\u6bcd", retry=True, retry_variant="incomplete_fragment"),
            'Translate exactly. If the source is incomplete, translate the incomplete fragment. Never ask for clarification.\nJapanese: "\u6bcd"\nEnglish:',
        )
        self.assertEqual(CAT_PROMPT_VERSION, "cat-translate-v6-fragment-strict-primary")

    def test_cat_num_predict_reads_bounded_environment_value(self) -> None:
        with patch.dict("os.environ", {"MANGA_CAT_NUM_PREDICT": "48"}):
            self.assertEqual(cat_num_predict(), 48)
        with patch.dict("os.environ", {"MANGA_CAT_NUM_PREDICT": "2"}):
            self.assertEqual(cat_num_predict(), 16)
        with patch.dict("os.environ", {"MANGA_CAT_NUM_PREDICT": "bad"}):
            self.assertEqual(cat_num_predict(), 48)

    def test_cat_second_retry_is_enabled_by_default_with_env_disable(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertTrue(cat_second_retry_enabled())
        with patch.dict("os.environ", {"MANGA_CAT_SECOND_RETRY": "0"}):
            self.assertFalse(cat_second_retry_enabled())

    def test_cat_bypass_reason_catches_short_and_noisy_sources(self) -> None:
        self.assertIsNone(cat_bypass_reason("\u4f55\u3042\u308c"))
        with patch.dict("os.environ", {"MANGA_CAT_BYPASS_RISKY_SOURCE": "1"}):
            self.assertEqual(cat_bypass_reason("\u4f55\u3042\u308c"), "short_ambiguous_fragment")
            self.assertEqual(cat_bypass_reason("\u25bd\u65e5\uff11\u56de\u5065\u4eba\u300e\u30a2\u30d9\u30c4\u30aa\u30ca"), "noisy_credit_or_metadata")
            self.assertEqual(cat_bypass_reason("\u30a6\u30a3\u30ea\u30a2\u30e0"), "short_ambiguous_fragment")
            self.assertEqual(cat_bypass_reason("\uff12\uff10\uff12\uff10\u5e74\uff11\uff12\u6708"), "noisy_credit_or_metadata")
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
            cat_reject_reason("\u6bcd", "I'm sorry, but I can't provide a translation of that text."),
            "cat_chatter",
        )
        self.assertEqual(
            cat_reject_reason("\u6bcd", "I don't understand the Japanese text. Could you please clarify?"),
            "cat_chatter",
        )
        self.assertEqual(
            cat_reject_reason("\u6bcd", "I'm ready to translate your text, but I need you to provide the Japanese passage."),
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
        self.assertEqual(cat_reject_reason("\u6bcd", "Here's a short English fragment: Indeed...!"), "cat_chatter")
        self.assertEqual(
            cat_reject_reason(
                "\u6bcd",
                'Sure, here\'s a translation of your Japanese sentence: "Mother."',
            ),
            "cat_chatter",
        )
        self.assertEqual(cat_reject_reason("\u6bcd", "Return only a short English fragment. Do not explain."), "cat_prompt_fragment")
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

    def test_short_kana_honorific_name_fragment_uses_source_shape(self) -> None:
        self.assertEqual(translate_kana_honorific_name_fragment("\u30ea\u30f3\u69d8"), "Rin sama")
        self.assertIsNone(translate_kana_honorific_name_fragment("\u30a2\u30fc\u30cb\u30e3\u69d8"))
        self.assertEqual(translate_kana_honorific_name_fragment("\u304b\u3050\u3084\u69d8\u2026"), "Kaguya sama...")
        self.assertIsNone(translate_kana_honorific_name_fragment("\u304a\u3070\u3055\u3093"))

    def test_finalize_cat_translation_overrides_short_kana_honorific_title(self) -> None:
        cleaned, final, reason = finalize_cat_translation(
            "\u304b\u3050\u3084\u69d8",
            "\u304b\u3050\u3084\u69d8",
            "Princess Kaguya",
            None,
        )

        self.assertEqual(cleaned, "Kaguya sama")
        self.assertEqual(final, "Kaguya sama")
        self.assertIsNone(reason)

    def test_honorific_postprocess_converts_generic_titles(self) -> None:
        result, changes = apply_honorific_title_replacements("\u304b\u3050\u3084\u69d8", "Princess Kaguya")

        self.assertEqual(result, "Kaguya sama")
        self.assertTrue(changes)

    def test_finalize_cat_translation_rejects_assistant_translation_chatter(self) -> None:
        source = "\u5fc5\u305a\u3001\u4eba\u6c17\u526f\u4f5c\uff23\u30ab\u30e9\u30fc\uff01\uff01\u3042\u306a\u305f\u3092\u3057\u305d\u306e\u601d\u3044\u51fa\u3055\u305b\u3066\u3001\u5f7c\u306f\u304d\u3063\u3068\u5f37\u304f\u306a\u308b\u3088\u3046\u306a\u6c17\u6301\u3061\u3060\u3002"
        cleaned, final, reason = finalize_cat_translation(
            source,
            source,
            'Sure, here\'s a translation of your Japanese sentence:\n\n"I definitely want to make you remember that color! It will surely give him the strength he needs."',
            None,
        )

        self.assertEqual(cleaned, "")
        self.assertEqual(final, "")
        self.assertEqual(reason, "cat_chatter")

    def test_finalize_cat_translation_can_salvage_explicit_english_answer(self) -> None:
        cleaned, final, reason = finalize_cat_translation(
            "\u78ba\u304b\u306b\uff0e\uff0e\uff0e\uff01",
            "\u78ba\u304b\u306b\uff0e\uff0e\uff0e\uff01",
            "Sure thing! Here's the translation for you:\nJapanese: \u78ba\u304b\u306b\u2026!\nEnglish: Indeed...!",
            None,
        )

        self.assertEqual(cleaned, "Indeed...!")
        self.assertEqual(final, "Indeed...!")
        self.assertIsNone(reason)

    def test_salvage_cat_translation_uses_explicit_english_marker_only(self) -> None:
        self.assertEqual(salvage_cat_translation("Japanese: 確かに…!\nEnglish: Indeed...!  This captures the sense."), "Indeed...!")
        self.assertEqual(salvage_cat_translation('It can be translated as "Oh my!" depending on context.'), "Oh my!")
        self.assertEqual(salvage_cat_translation('It means "\u3042\u3063\u304b\u308c\u3093!" in Japanese.'), "")
        self.assertEqual(
            salvage_cat_translation('It appears to be a typo. It can be translated as "time for the decision."'),
            "",
        )
        self.assertEqual(
            salvage_cat_translation('If you mean fuel, it would be expressed as "I have gas."'),
            "",
        )
        self.assertEqual(
            salvage_cat_translation(
                "The Japanese phrase \"\u76ee\u7acb\u3063\u3066\u306f\u3044\u3051\u306a\u3044\" translates to: **\"You shouldn't stand out.\"**",
                source_text="\u76ee\u7acb\u3063\u3066\u306f\u3044\u3051\u306a\u3044",
            ),
            "You shouldn't stand out.",
        )
        self.assertEqual(
            salvage_cat_translation(
                "The Japanese phrase \"\u30ac\u30b9\u304c\u3044\u3066\" translates to \"Is there gas?\"",
                source_text="\u30ac\u30b9\u304c\u3044\u3066",
            ),
            "",
        )
        self.assertEqual(
            salvage_cat_translation(
                'The Japanese phrase "\u307e\u305f\u982d\u304b\u3089\u304b\u3051\u307e\u3059\u3088." means "I\'ll start over from the beginning again."',
                source_text="\u307e\u305f\u982d\u304b\u3089\u304b\u3051\u307e\u3059\u3088\u3002",
            ),
            "I'll start over from the beginning again.",
        )

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

    def test_cat_suspected_bad_translation_uses_strict_second_pass(self) -> None:
        class FakeCat:
            def __init__(self) -> None:
                self.debug = {}

            def retry_translation(self, text):
                raise AssertionError("normal retry should not be used for accepted suspected translations")

            def second_retry_translation(self, text, previous_translation=""):
                self.debug[text] = {
                    "cat_suspect_second_pass_attempted": True,
                    "cat_suspect_second_pass_accepted": True,
                    "cat_suspect_second_pass_previous_translation": previous_translation,
                    "cat_suspect_second_pass_raw_translation": "You should not stand out...",
                    "cat_suspect_second_pass_cleaned_translation": "You should not stand out...",
                    "cat_suspect_second_pass_final": "You should not stand out...",
                    "cat_suspect_second_pass_rejected": False,
                    "cat_suspect_second_pass_reject_reason": "",
                    "cat_rejected": False,
                    "cat_reject_reason": "",
                    "cat_final": "You should not stand out...",
                }
                return "You should not stand out..."

            def debug_info_for(self, text, debug_id=None):
                return self.debug.get(text, {})

        block = TextBlock("\u76ee\u7acb\u3063\u3066\u306f\u3044\u3051\u306a\u3044", (0, 0, 10, 10), 90)
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
            page_order_report=[{"line_id": "line-1", "source_text": block.text, "page_order": 1}],
            translations={"line-1": "Could you please provide the Japanese passage?"},
            translation_contexts={
                "line-1": {
                    "primary_translator": "cat",
                    "cat_used": True,
                    "cat_rejected": False,
                    "cat_reject_reason": "",
                }
            },
        )

        attempted, accepted = retry_cat_failures(page, FakeCat(), PipelineConfig(translator="cat"))

        self.assertEqual((attempted, accepted), (1, 1))
        self.assertEqual(page.translations["line-1"], "You should not stand out...")
        self.assertEqual(page.translation_contexts["line-1"]["cat_retry_mode"], "suspected_bad")
        self.assertTrue(page.translation_contexts["line-1"]["cat_suspect_second_pass_attempted"])
        self.assertTrue(page.translation_contexts["line-1"]["cat_suspect_second_pass_accepted"])

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
