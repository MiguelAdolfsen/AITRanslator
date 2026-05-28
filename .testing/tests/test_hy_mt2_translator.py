import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from manga_local_translator.config import PipelineConfig
from manga_local_translator.hf_translators import (
    HY_MT2_PROMPT_VERSION,
    build_hy_mt2_prompt,
    clean_hy_mt2_output,
    ensure_hy_mt2_weight_alias,
    hy_mt2_num_predict,
    resolve_hy_mt2_model_name,
)
from manga_local_translator.pipeline import translation_cache_stage
from manga_local_translator.translate import build_translator


class HyMt2TranslatorTests(unittest.TestCase):
    def test_build_hy_mt2_prompt_uses_official_output_constraint(self) -> None:
        prompt = build_hy_mt2_prompt("\u6bcd")

        self.assertIn("Translate the following text into English", prompt)
        self.assertIn("only output the translated result", prompt)
        self.assertTrue(prompt.endswith("\u6bcd"))

    def test_build_hy_mt2_prompt_includes_context_and_visual_facts(self) -> None:
        prompt = build_hy_mt2_prompt(
            "\u5f85\u3063\u3066",
            context_lines=("\u524d\u306e\u30bb\u30ea\u30d5", "\u6b21\u306e\u30bb\u30ea\u30d5"),
            visual_facts=("speaker is surprised",),
        )

        self.assertIn("[Background Information]", prompt)
        self.assertIn("\u524d\u306e\u30bb\u30ea\u30d5", prompt)
        self.assertIn("Visual fact: speaker is surprised", prompt)
        self.assertIn("[Source Text]\n\u5f85\u3063\u3066", prompt)

    def test_clean_hy_mt2_output_removes_common_label(self) -> None:
        self.assertEqual(clean_hy_mt2_output('English: "Mom"'), "Mom")

    def test_hy_mt2_num_predict_env_override_is_bounded(self) -> None:
        with patch.dict(os.environ, {"MANGA_HY_MT2_NUM_PREDICT": "2"}):
            self.assertEqual(hy_mt2_num_predict(), 16)
        with patch.dict(os.environ, {"MANGA_HY_MT2_NUM_PREDICT": "bad"}):
            self.assertEqual(hy_mt2_num_predict(), 128)

    def test_build_translator_accepts_hy_mt2_model_override(self) -> None:
        sentinel = object()
        with patch("manga_local_translator.translate.HyMt2Translator", return_value=sentinel) as translator:
            result = build_translator("hy-mt2", hy_mt2_model_name="local/hy-mt2")

        self.assertIs(result, sentinel)
        translator.assert_called_once_with(model_name="local/hy-mt2", glossary_path=None)

    def test_hy_mt2_cache_stage_includes_model_and_prompt_identity(self) -> None:
        first = translation_cache_stage(PipelineConfig(translator="hy-mt2", hy_mt2_model_name="model-a"), "primary")
        second = translation_cache_stage(PipelineConfig(translator="hy-mt2", hy_mt2_model_name="model-b"), "primary")

        self.assertIn(HY_MT2_PROMPT_VERSION, first)
        self.assertNotEqual(first, second)

    def test_hy_mt2_local_folder_gets_model_safetensors_alias(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            source = model_dir / "MT2-1.8B.safetensors"
            source.write_bytes(b"weights")

            ensure_hy_mt2_weight_alias(model_dir)

            self.assertTrue((model_dir / "model.safetensors").exists())

    def test_resolve_hy_mt2_model_name_preserves_remote_id(self) -> None:
        self.assertEqual(resolve_hy_mt2_model_name("example/remote-model"), "example/remote-model")


if __name__ == "__main__":
    unittest.main()
