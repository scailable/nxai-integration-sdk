"""Tests for the no-response processor's side effect (logging only, no reply).

Mirrors the construction style used in nxai-utilities/tests/test_message_roundtrip.py:
build an ObjectsPostProcessorMessage directly (no real IPC channel) and assert on
what handle_message logs and returns.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nxai-utilities"))

from nxai_postprocessor_sdk import ObjectsPostProcessorMessage, ObjectMetadata, EventMetadata

from importlib import import_module

noresponse_example = import_module("postprocessor-python-noresponse-example")


class TestHandleMessage(unittest.TestCase):
    def test_returns_nothing_and_logs_object_and_event_counts(self):
        msg = ObjectsPostProcessorMessage()
        msg.device_id = "device-42"
        msg.objects["person"] = [ObjectMetadata()]
        msg.events.append(EventMetadata())

        with self.assertLogs(noresponse_example.logger, level="INFO") as captured:
            result = noresponse_example.handle_message(msg)

        self.assertIsNone(result)
        self.assertTrue(
            any("1 object classes, 1 events" in record.getMessage() for record in captured.records)
        )

    def test_handles_empty_message_without_error(self):
        msg = ObjectsPostProcessorMessage()
        msg.device_id = "device-empty"

        with self.assertLogs(noresponse_example.logger, level="INFO") as captured:
            result = noresponse_example.handle_message(msg)

        self.assertIsNone(result)
        self.assertTrue(
            any("0 object classes, 0 events" in record.getMessage() for record in captured.records)
        )


EXIT = object()  # Sentinel script item: simulates IpcSignal.EXIT via exit_requested.
UNPARSEABLE = b"unparseable-frame"  # Script item for which parse_objects_message returns None.


class _ScriptExhausted(BaseException):
    """Raised when the loop asks for more messages than the script holds.

    Derives from BaseException rather than Exception on purpose. run_message_loop wraps
    its body in `except Exception: break`, which would otherwise swallow this guard and
    exit the loop normally -- turning every assertion below into one that cannot fail.
    An EXIT-swallowing regression, exactly the defect these tests exist to catch, would
    then still pass.
    """


class _FakeProcessor:
    """Replays a scripted sequence of wait_for_message() outcomes.

    `calls` counts wait_for_message() invocations, which is what pins down where the
    loop stopped: asserting on exit_requested alone cannot distinguish "returned on
    EXIT" from "kept spinning past it".
    """

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0
        self.exit_requested = False

    def wait_for_message(self, timeout_ms=None):
        if not self._script:
            raise _ScriptExhausted("wait_for_message called past the end of the script")
        self.calls += 1
        item = self._script.pop(0)
        if item is EXIT:
            self.exit_requested = True
            return None
        if isinstance(item, Exception):
            raise item
        return item

    def parse_objects_message(self, message_bytes):
        if message_bytes == UNPARSEABLE:
            return None
        msg = ObjectsPostProcessorMessage()
        msg.device_id = "device-from-script"
        return msg


class TestMessageLoopSurvivability(unittest.TestCase):
    """The engine keeps sending frames; a processor that exits is dropped."""

    def test_exit_signal_ends_the_loop_cleanly(self):
        # A graceful EXIT must return, not be swallowed as a plain timeout that
        # keeps the loop spinning until the AI Manager kills the process. The call
        # count is the assertion that catches the swallow: a loop that treats EXIT
        # as a timeout asks for a second message.
        processor = _FakeProcessor([EXIT])

        with self.assertLogs(noresponse_example.logger, level="INFO") as captured:
            noresponse_example.run_message_loop(processor)

        self.assertEqual(1, processor.calls, "loop must return on EXIT, not ask for more")
        self.assertTrue(
            any("graceful exit signal" in record.getMessage() for record in captured.records)
        )

    def test_timeout_does_not_end_the_loop(self):
        processor = _FakeProcessor([None, None, b"frame-bytes", EXIT])

        with self.assertLogs(noresponse_example.logger, level="INFO") as captured:
            noresponse_example.run_message_loop(processor)

        self.assertEqual(4, processor.calls, "loop must survive both timeouts and the frame")
        self.assertTrue(processor.exit_requested)
        # The one parseable frame was handled: a loop that skipped it would still
        # reach EXIT with the same call count.
        self.assertTrue(
            any("device-from-script" in record.getMessage() for record in captured.records)
        )

    def test_unparseable_frame_does_not_end_the_loop(self):
        # A frame the SDK cannot parse is logged and skipped, not fatal.
        processor = _FakeProcessor([UNPARSEABLE, b"frame-bytes", EXIT])

        noresponse_example.run_message_loop(processor)

        self.assertEqual(3, processor.calls, "loop must survive an unparseable frame")
        self.assertTrue(processor.exit_requested)

    def test_raising_message_ends_the_loop(self):
        # Unlike a plain timeout, an unexpected exception from wait_for_message
        # is not survived: the loop's except clause breaks rather than continues.
        processor = _FakeProcessor([RuntimeError("transport hiccup"), EXIT])

        noresponse_example.run_message_loop(processor)

        self.assertEqual(1, processor.calls, "loop must break on the exception, before EXIT")
        self.assertFalse(processor.exit_requested)


if __name__ == "__main__":
    unittest.main()
