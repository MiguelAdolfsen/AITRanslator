from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from manga_local_translator.translation_rules import (
    load_translation_glossary,
    normalize_japanese_for_translation,
    postprocess_translation,
    prepare_source_for_translation,
    translate_known_phrase,
    translation_debug_info,
)


class TranslationRulesTests(unittest.TestCase):
    def tearDown(self) -> None:
        load_translation_glossary.cache_clear()

    def test_normalize_and_phrasebook_handle_common_sfx(self) -> None:
        self.assertEqual(normalize_japanese_for_translation(" く っ "), "くっ")
        self.assertEqual(translate_known_phrase("くっ"), "Ngh.")
        self.assertEqual(translate_known_phrase("なっ．．．！？"), "Wha...!?")
        self.assertEqual(translate_known_phrase("ドキドキ"), "Thump thump.")
        self.assertEqual(translate_known_phrase("たたた"), "Tap tap tap.")
        self.assertEqual(translate_known_phrase("ポリポリボリ"), "Crunch crunch.")
        self.assertEqual(translate_known_phrase("\uff0e\uff0e\uff0e\u305d\u3093\u306a"), "...No way.")
        self.assertEqual(translate_known_phrase("\u6b8b\u5ff5\u306a"), "What a shame.")

    def test_phrasebook_handles_common_sasuga_senpai_phrase(self) -> None:
        self.assertEqual(translate_known_phrase("さすがっす先輩"), "A-as expected, senpai...!")

    def test_source_cleanup_normalizes_common_ocr_errors(self) -> None:
        prepared, replacements = prepare_source_for_translation("おかしやさんとわるもの")

        self.assertIn("お菓子屋さん", prepared)
        self.assertIn("悪者", prepared)
        self.assertIn({"source": "おかしやさん", "target": "お菓子屋さん"}, replacements)

    def test_honorifics_are_preserved_as_suffixes(self) -> None:
        prepared, replacements = prepare_source_for_translation("アーニャさん")

        self.assertEqual(prepared, "アーニャ san")
        self.assertEqual(replacements, [{"source": "アーニャさん", "target": "アーニャ san"}])
        self.assertEqual(postprocess_translation("アーニャさん", prepared, "Mr. Anya"), "Anya san")

    def test_postprocess_removes_unnecessary_gendering_for_omaera(self) -> None:
        source = "\u304a\u307e\u3048\u3089"

        self.assertEqual(postprocess_translation(source, source, "Come on, boys!"), "Come on, you guys!")

    def test_postprocess_compacts_common_verbose_phrases(self) -> None:
        self.assertEqual(
            postprocess_translation("\u3044\u304f", "\u3044\u304f", "I'm going to go save them."),
            "I'll go save them.",
        )
        self.assertEqual(
            postprocess_translation("\u3044\u3051\u308b", "\u3044\u3051\u308b", "You'll be able to do it."),
            "you can do it.",
        )

    def test_translation_debug_reports_source_and_target_replacements(self) -> None:
        debug = translation_debug_info("アーニャさん", "Mr. Anya")

        self.assertEqual(debug["prepared_source"], "アーニャ san")
        self.assertEqual(debug["glossary_target_replacements"][0]["target"], "Anya san")

    def test_custom_glossary_exact_phrase_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "translation_glossary.json"
            path.write_text(
                json.dumps({"exact_phrases": {"秘密": "classified"}}, ensure_ascii=False),
                encoding="utf-8",
            )

            glossary = load_translation_glossary(path)

        self.assertEqual(translate_known_phrase("秘密", glossary), "classified")


if __name__ == "__main__":
    unittest.main()
