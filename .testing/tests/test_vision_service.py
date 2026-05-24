from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np

from manga_local_translator.config import PipelineConfig
from manga_local_translator.detect_types import TextBlock
from manga_local_translator.render import TextFit, RenderLayout
from manga_local_translator.vision_artifact import create_vision_artifact
from manga_local_translator import vision_service
from manga_local_translator.vision_service import apply_vision_facts, apply_vision_repair, build_vision_requests, get_qwen_vision_client


class FakeVisionClient:
    model_name = "fake-vision"

    def __init__(self, response: str, json_repair_response: str = "") -> None:
        self.response = response
        self.json_repair_response = json_repair_response
        self.prompt = ""
        self.json_repair_prompt = ""
        self.image_path = None

    def repair(self, prompt: str, image_path: Path) -> str:
        self.prompt = prompt
        self.image_path = image_path
        return self.response

    def extract_facts(self, prompt: str, image_path: Path) -> str:
        self.prompt = prompt
        self.image_path = image_path
        return self.response

    def repair_json(self, prompt: str) -> str:
        self.json_repair_prompt = prompt
        return self.json_repair_response


class VisionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.image = np.full((120, 120, 3), 255, dtype=np.uint8)
        self.block = TextBlock("母", (10, 10, 40, 50), 0.9, "ctd", {})
        self.layout = RenderLayout(source_box=self.block.box, render_box=(10, 10, 60, 60), bubble_box=(5, 5, 70, 70))
        self.fit = TextFit(12, 1, "fit", False, 20, 14, 40, 40, 12, 14)
        self.page_order = [{"page_order": 1, "box": self.block.box, "source_text": self.block.text, "detector": "ctd"}]

    def tearDown(self) -> None:
        vision_service._VISION_CLIENT_CACHE.clear()

    def test_qwen_vision_client_is_reused_for_same_config(self) -> None:
        created = []

        class FakeCachedClient:
            def __init__(self, *, model_path=None, projector_path=None) -> None:
                self.model_path = model_path
                self.projector_path = projector_path
                created.append(self)

        original = vision_service.QwenVisionClient
        try:
            vision_service.QwenVisionClient = FakeCachedClient
            with tempfile.TemporaryDirectory() as temp:
                model = Path(temp) / "model.gguf"
                projector = Path(temp) / "mmproj.gguf"
                model.write_text("", encoding="utf-8")
                projector.write_text("", encoding="utf-8")
                config = PipelineConfig(vision_enabled=True, qwen_model_path=model, vision_projector_path=projector)

                first = get_qwen_vision_client(config)
                second = get_qwen_vision_client(config)
        finally:
            vision_service.QwenVisionClient = original

        self.assertIs(first, second)
        self.assertEqual(len(created), 1)

    def test_suspicious_trigger_selects_bad_translation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            artifact = create_vision_artifact(self.image, [self.block], {self.block.text: "..."}, self.page_order, Path(temp) / "out.png", mode="numbered_page")

            requests = build_vision_requests(
                [self.block],
                {self.block.text: "..."},
                {self.block.text: {}},
                artifact,
                [self.layout],
                [self.fit],
                image_width=120,
                image_height=120,
                trigger="suspicious",
            )

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].issue, "suspected_bad_translation")

    def test_suspicious_trigger_selects_low_confidence_short_ocr(self) -> None:
        block = TextBlock("\u3046\u3093", (10, 10, 40, 50), 0.3, "ctd", {})
        with tempfile.TemporaryDirectory() as temp:
            artifact = create_vision_artifact(self.image, [block], {block.text: "Yeah."}, self.page_order, Path(temp) / "out.png", mode="numbered_page")

            requests = build_vision_requests(
                [block],
                {block.text: "Yeah."},
                {block.text: {}},
                artifact,
                [self.layout],
                [self.fit],
                image_width=120,
                image_height=120,
                trigger="suspicious",
            )

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].issue, "low_confidence_short_ocr")

    def test_layout_trigger_selects_layout_warning_only(self) -> None:
        fit = TextFit(7, 5, "clipped", True, 80, 80, 20, 20, 12, 14, overflow_width=4, overflow_height=5)
        with tempfile.TemporaryDirectory() as temp:
            artifact = create_vision_artifact(self.image, [self.block], {self.block.text: "Mom"}, self.page_order, Path(temp) / "out.png", mode="numbered_page")

            requests = build_vision_requests(
                [self.block],
                {self.block.text: "Mom"},
                {self.block.text: {}},
                artifact,
                [self.layout],
                [fit],
                image_width=120,
                image_height=120,
                trigger="layout",
            )

        self.assertEqual(len(requests), 1)
        self.assertIn("layout:text_clipped", requests[0].issue)

    def test_valid_vision_repair_updates_translation_and_context(self) -> None:
        response = (
            '{"lines":[{"line_id":"out_001","number":1,"action":"replace","translation":"Mom",'
            '"speaker":"woman in panel","situation":"speaking softly",'
            '"visual_evidence":"speaker appears to be the mother","confidence":"high","warnings":[]}]}'
        )
        client = FakeVisionClient(response)
        translations = {self.block.text: "..."}
        contexts = {self.block.text: {}}
        with tempfile.TemporaryDirectory() as temp:
            artifact = apply_vision_repair(
                image_bgr=self.image,
                blocks=[self.block],
                translations=translations,
                translation_contexts=contexts,
                page_order_report=self.page_order,
                render_layouts=[self.layout],
                render_fits=[self.fit],
                output_path=Path(temp) / "out.png",
                config=PipelineConfig(vision_enabled=True, vision_trigger="suspicious"),
                client=client,
            )

        self.assertIsNotNone(artifact)
        self.assertEqual(translations[self.block.text], "Mom")
        self.assertTrue(contexts[self.block.text]["vision_accepted"])
        self.assertEqual(contexts[self.block.text]["vision_trigger_reason"], "suspected_bad_translation")
        self.assertEqual(contexts[self.block.text]["vision_final_source"], "vision")
        self.assertIn("Do not OCR", client.prompt)

    def test_valid_vision_facts_update_context_without_translation(self) -> None:
        response = json.dumps(
            {
                "lines": [
                    {
                        "line_id": "out_001",
                        "number": 1,
                        "source_text": self.block.text,
                        "bubble_type": "speech",
                        "speaker_position": "woman on left",
                        "visible_emotion": "calm",
                        "observable_action": "facing right",
                        "mapping_confidence": "high",
                        "facts": ["speech bubble near woman on left"],
                        "needs_review": False,
                        "risk_flags": [],
                        "warnings": [],
                    }
                ]
            },
            ensure_ascii=False,
        )
        contexts = {self.block.text: {}}
        client = FakeVisionClient(response)
        with tempfile.TemporaryDirectory() as temp:
            artifact = apply_vision_facts(
                image_bgr=self.image,
                blocks=[self.block],
                translation_contexts=contexts,
                page_order_report=self.page_order,
                output_path=Path(temp) / "out.png",
                config=PipelineConfig(vision_facts_enabled=True, debug=True),
                client=client,
            )

        self.assertIsNotNone(artifact)
        self.assertEqual(artifact.image_path.name, "out.vision-facts.png")
        self.assertTrue(contexts[self.block.text]["vision_facts_accepted"])
        self.assertEqual(contexts[self.block.text]["bubble_type"], "speech")
        self.assertEqual(contexts[self.block.text]["visual_facts"], ["speech bubble near woman on left"])
        self.assertIn("Do not translate", client.prompt)

    def test_invalid_vision_facts_are_not_accepted(self) -> None:
        response = json.dumps(
            {
                "lines": [
                    {
                        "line_id": "out_001",
                        "number": 1,
                        "source_text": self.block.text,
                        "translation": "Mom",
                        "mapping_confidence": "high",
                        "facts": ["Mom."],
                    }
                ]
            },
            ensure_ascii=False,
        )
        contexts = {self.block.text: {}}
        with tempfile.TemporaryDirectory() as temp:
            apply_vision_facts(
                image_bgr=self.image,
                blocks=[self.block],
                translation_contexts=contexts,
                page_order_report=self.page_order,
                output_path=Path(temp) / "out.png",
                config=PipelineConfig(vision_facts_enabled=True),
                client=FakeVisionClient(response),
            )

        self.assertFalse(contexts[self.block.text]["vision_facts_accepted"])
        self.assertEqual(contexts[self.block.text]["vision_facts_reject_reason"], "contains_translation_field")

    def test_invalid_vision_repair_keeps_existing_translation(self) -> None:
        response = (
            '{"lines":[{"line_id":"out_001","number":1,"action":"replace","translation":"母",'
            '"speaker":null,"situation":null,"visual_evidence":"not visually informed",'
            '"confidence":"high","warnings":[]}]}'
        )
        translations = {self.block.text: "Mom"}
        contexts = {self.block.text: {"qwen_rejected": True}}
        with tempfile.TemporaryDirectory() as temp:
            apply_vision_repair(
                image_bgr=self.image,
                blocks=[self.block],
                translations=translations,
                translation_contexts=contexts,
                page_order_report=self.page_order,
                render_layouts=[self.layout],
                render_fits=[self.fit],
                output_path=Path(temp) / "out.png",
                config=PipelineConfig(vision_enabled=True, vision_trigger="suspicious"),
                client=FakeVisionClient(response),
            )

        self.assertEqual(translations[self.block.text], "Mom")
        self.assertFalse(contexts[self.block.text]["vision_accepted"])
        self.assertEqual(contexts[self.block.text]["vision_reject_reason"], "contains_japanese")
        self.assertEqual(contexts[self.block.text]["vision_final_source"], "text")

    def test_malformed_vision_json_can_be_repaired_without_resending_image(self) -> None:
        repair_response = (
            '{"lines":[{"line_id":"out_001","number":1,"action":"replace","translation":"Mom",'
            '"speaker":"woman in panel","situation":"speaking softly",'
            '"visual_evidence":"speaker appears to be the mother","confidence":"high","warnings":[]}]}'
        )
        client = FakeVisionClient("{not valid json", json_repair_response=repair_response)
        translations = {self.block.text: "..."}
        contexts = {self.block.text: {}}
        with tempfile.TemporaryDirectory() as temp:
            apply_vision_repair(
                image_bgr=self.image,
                blocks=[self.block],
                translations=translations,
                translation_contexts=contexts,
                page_order_report=self.page_order,
                render_layouts=[self.layout],
                render_fits=[self.fit],
                output_path=Path(temp) / "out.png",
                config=PipelineConfig(vision_enabled=True, vision_trigger="suspicious"),
                client=client,
            )

        self.assertEqual(translations[self.block.text], "Mom")
        self.assertTrue(contexts[self.block.text]["vision_accepted"])
        self.assertTrue(contexts[self.block.text]["vision_json_repair_attempted"])
        self.assertTrue(contexts[self.block.text]["vision_json_repair_used"])
        self.assertIn("JSON repair task only", client.json_repair_prompt)
        self.assertIn("out_001", contexts[self.block.text]["vision_raw_json_repair_response"])

    def test_json_repair_is_ignored_when_it_does_not_improve_structure(self) -> None:
        client = FakeVisionClient("{not valid json", json_repair_response="still invalid")
        translations = {self.block.text: "Mom"}
        contexts = {self.block.text: {"qwen_rejected": True}}
        with tempfile.TemporaryDirectory() as temp:
            apply_vision_repair(
                image_bgr=self.image,
                blocks=[self.block],
                translations=translations,
                translation_contexts=contexts,
                page_order_report=self.page_order,
                render_layouts=[self.layout],
                render_fits=[self.fit],
                output_path=Path(temp) / "out.png",
                config=PipelineConfig(vision_enabled=True, vision_trigger="suspicious"),
                client=client,
            )

        self.assertEqual(translations[self.block.text], "Mom")
        self.assertFalse(contexts[self.block.text]["vision_accepted"])
        self.assertTrue(contexts[self.block.text]["vision_json_repair_attempted"])
        self.assertFalse(contexts[self.block.text]["vision_json_repair_used"])
        self.assertEqual(contexts[self.block.text]["vision_reject_reason"], "missing_line_id")


if __name__ == "__main__":
    unittest.main()
