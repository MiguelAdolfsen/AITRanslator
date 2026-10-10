from __future__ import annotations

import logging
import unittest

from manga_local_translator.gui import (
    SETTING_CHOICES,
    SETTING_DEFAULTS,
    _QueueLogHandler,
    _QueueWriter,
    format_duration,
)


class GuiHelperTests(unittest.TestCase):
    def test_format_duration(self) -> None:
        self.assertEqual(format_duration(0), "00:00")
        self.assertEqual(format_duration(75.9), "01:15")
        self.assertEqual(format_duration(3725), "1:02:05")

    def test_queue_writer_emits_complete_lines(self) -> None:
        events: list[tuple] = []
        writer = _QueueWriter(events.append)
        writer.write("Downloading model...\nHalf")
        self.assertEqual(events, [("line", "Downloading model...", "info")])
        writer.write(" done\r\n\n")
        writer.write("tail")
        writer.flush()
        self.assertEqual(
            events,
            [
                ("line", "Downloading model...", "info"),
                ("line", "Half done", "info"),
                ("line", "tail", "info"),
            ],
        )

    def test_log_handler_forwards_info_and_skips_gui_records(self) -> None:
        events: list[tuple] = []
        handler = _QueueLogHandler(events.append)
        pipeline_logger = logging.getLogger("manga_local_translator.test_gui_helpers")
        pipeline_logger.setLevel(logging.DEBUG)
        pipeline_logger.propagate = False
        pipeline_logger.addHandler(handler)
        try:
            pipeline_logger.info("Page %d/%d", 2, 5)
            pipeline_logger.debug("noise")
        finally:
            pipeline_logger.removeHandler(handler)
        gui_record = logging.LogRecord("manga_local_translator.gui", logging.INFO, __file__, 1, "GUI initialized", (), None)
        handler.handle(gui_record)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0][:3], ("log", logging.INFO, "Page 2/5"))

    def test_choice_defaults_are_valid_options(self) -> None:
        for key, choices in SETTING_CHOICES.items():
            self.assertIn(SETTING_DEFAULTS[key], {choice.value for choice in choices}, key)


if __name__ == "__main__":
    unittest.main()
