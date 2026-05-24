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
        self.assertEqual(translate_known_phrase("ドキドキ"), "Thump thump.")
        self.assertEqual(translate_known_phrase("たたた"), "Tap tap tap.")
        self.assertEqual(translate_known_phrase("ポリポリボリ"), "Crunch crunch.")

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

    def test_postprocess_fixes_kotonaki_idiom(self) -> None:
        source = "\u30d5\u30a9\u30fc\u30b8\u30e3\u30fc\u5bb6\u306f\u4e8b\u306a\u304d\u3092\u5f97\u305f"

        self.assertEqual(
            postprocess_translation(source, source, "The Forger family has won everything for this smile!"),
            "The Forger family got through without incident for this smile!",
        )

    def test_postprocess_compacts_common_verbose_phrases(self) -> None:
        self.assertEqual(
            postprocess_translation("いく", "いく", "I'm going to go save them."),
            "I'll go save them.",
        )

    def test_postprocess_cleans_broad_bad_patterns(self) -> None:
        cases = [
            (
                "\u7d44\u7e54",
                "\u7d44\u7e54",
                "If you do that, you'll be able to watch my organization for what it is.",
                "If you do that, you can be recognized as a member of my organization.",
            ),
            (
                "\u7acb\u6d3e\u306a\u5b50\u5206",
                "\u7acb\u6d3e\u306a\u5b50\u5206",
                "You're both a good son of a bitch.",
                "You're both a good subordinate.",
            ),
            (
                "\u30a2\u30fc\u30cb\u30e3\u3055\u3093",
                "\u30a2\u30fc\u30cb\u30e3 san",
                "Anja san liked it.",
                "Anya san liked it.",
            ),
        ]

        for source, prepared, translated, expected in cases:
            with self.subTest(translated=translated):
                self.assertEqual(postprocess_translation(source, prepared, translated), expected)

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
