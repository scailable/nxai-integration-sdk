"""Unit tests for the image-preprocessor business logic.

External behavior only: a built ImagePreprocessorMessage goes in, and the
mirrored (or passed-through) image is asserted on the way out, at unchanged
resolution. No IPC, no sockets. The mirror flag is read fresh from
`msg.user_settings["mirrorimage"]` on every frame, falling back
to `DEFAULT_MIRROR_IMAGE` when the engine forwards no settings.

Every frame here carries the payload this preprocessor actually expects: uint8
RGB at source resolution, one byte per element. Pillow does the actual pixel
move (`ImageOps.mirror`); these tests exist to catch a layout bug -- CHW vs.
HWC, or a channel/row mixed up in the numpy reshape/transpose around it -- not
to re-verify Pillow's own mirror.
"""

import importlib.util
import os
import sys
import unittest

script_dir = os.path.dirname(os.path.abspath(__file__))
# nxai_postprocessor_sdk lives one level up, in nxai-utilities/
sys.path.insert(0, os.path.join(script_dir, "../nxai-utilities"))

from nxai_postprocessor_sdk import ImageData, ImageOrder, ImagePreprocessorMessage

spec = importlib.util.spec_from_file_location(
    "preprocessor_python_image_example",
    os.path.join(script_dir, "preprocessor-python-image-example.py"),
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _message(height, width, values, layout=ImageOrder.CHW, user_settings=None):
    image = ImageData()
    image.height = height
    image.width = width
    image.bytes = bytes(values)
    image.layout = layout
    msg = ImagePreprocessorMessage()
    msg.image = image
    msg.user_settings = user_settings or {}
    return msg


class TestHandleMessage(unittest.TestCase):
    def test_mirrors_at_full_resolution_with_default_when_no_settings_forwarded(self):
        # 2x2 frame, 3 channels, CHW: three planes of two rows of two elements.
        msg = _message(2, 2, list(range(12)))
        module.handle_message(msg)
        self.assertEqual(msg.image.width, 2)
        self.assertEqual(msg.image.height, 2)
        self.assertEqual(
            list(msg.image.bytes),
            [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10],
        )

    def test_mirrors_a_single_row_chw_frame(self):
        # A 1x2 frame, one distinct value per pixel per channel.
        msg = _message(1, 2, [1, 2, 3, 4, 5, 6])
        module.handle_message(msg)
        self.assertEqual(list(msg.image.bytes), [2, 1, 4, 3, 6, 5])

    def test_mirrors_chw_across_several_planes_and_rows(self):
        # 2x3 frame (w=3, h=2), 3 channels: each plane is 2 rows of 3 elements.
        # Only elements within a row move -- rows keep their place in a plane,
        # and planes keep their place in the buffer.
        msg = _message(2, 3, list(range(18)))
        module.handle_message(msg)
        self.assertEqual(
            list(msg.image.bytes),
            [2, 1, 0, 5, 4, 3, 8, 7, 6,
             11, 10, 9, 14, 13, 12, 17, 16, 15],
        )

    def test_mirrors_hwc_pixels_without_scrambling_channels(self):
        # 1x2 frame, 3 interleaved channels per pixel: pixel 0 is (1, 2, 3),
        # pixel 1 is (4, 5, 6). The pixels swap; each pixel's channels do not.
        msg = _message(1, 2, [1, 2, 3, 4, 5, 6], layout=ImageOrder.HWC)
        module.handle_message(msg)
        self.assertEqual(list(msg.image.bytes), [4, 5, 6, 1, 2, 3])

    def test_passthrough_when_disabled_via_forwarded_settings(self):
        msg = _message(1, 2, [1, 2, 3, 4, 5, 6], user_settings={"mirrorimage": "false"})
        original_bytes = msg.image.bytes
        module.handle_message(msg)
        self.assertEqual(msg.image.bytes, original_bytes)

    def test_tolerates_non_string_wire_value(self):
        # A native msgpack bool would otherwise raise AttributeError from `.lower()`
        # called straight on the value; str(value).lower() must handle it instead of
        # crashing on the first frame.
        msg = _message(1, 2, [1, 2, 3, 4, 5, 6], user_settings={"mirrorimage": True})
        module.handle_message(msg)
        self.assertEqual(list(msg.image.bytes), [2, 1, 4, 3, 6, 5])


EXIT = object()  # Sentinel script item: simulates IpcSignal.EXIT via exit_requested.
UNPARSEABLE = b"unparseable-frame"  # Script item for which parse_image_message returns None.


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
    EXIT" from "kept spinning past it". `sent` collects the serialized responses, so a
    frame that was received but never mirrored and returned is visible too.
    """

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0
        self.exit_requested = False
        self.sent = []

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

    def parse_image_message(self, message_bytes):
        if message_bytes == UNPARSEABLE:
            return None
        return _message(1, 2, [1, 2, 3, 4, 5, 6])

    def write_image_message(self, msg):
        # Stand in for msgpack serialization: hand back the mirrored message so the
        # test can assert the frame really went through handle_message.
        return msg

    def send_response(self, response_bytes):
        self.sent.append(response_bytes)
        return True


class TestMessageLoopSurvivability(unittest.TestCase):
    """The engine keeps sending frames; a processor that exits is dropped."""

    def test_exit_signal_ends_the_loop_cleanly(self):
        # A graceful EXIT must return, not be swallowed as a plain timeout that
        # keeps the loop spinning until the AI Manager kills the process. The call
        # count is the assertion that catches the swallow: a loop that treats EXIT
        # as a timeout asks for a second message.
        processor = _FakeProcessor([EXIT])

        with self.assertLogs(module.logger, level="INFO") as captured:
            module.run_message_loop(processor)

        self.assertEqual(1, processor.calls, "loop must return on EXIT, not ask for more")
        self.assertTrue(
            any("graceful exit signal" in record.getMessage() for record in captured.records)
        )

    def test_timeout_does_not_end_the_loop(self):
        processor = _FakeProcessor([None, None, b"frame-bytes", EXIT])

        module.run_message_loop(processor)

        self.assertEqual(4, processor.calls, "loop must survive both timeouts and the frame")
        self.assertTrue(processor.exit_requested)
        # The one frame was actually handled: a loop that skipped it would still
        # reach EXIT with the same call count.
        self.assertEqual(1, len(processor.sent))
        mirrored = processor.sent[0]
        self.assertEqual(list(mirrored.image.bytes), [2, 1, 4, 3, 6, 5])

    def test_unparseable_frame_does_not_end_the_loop(self):
        # A frame the SDK cannot parse is logged and skipped, not fatal.
        processor = _FakeProcessor([UNPARSEABLE, b"frame-bytes", EXIT])

        module.run_message_loop(processor)

        self.assertEqual(3, processor.calls, "loop must survive an unparseable frame")
        self.assertEqual(1, len(processor.sent), "only the parseable frame is answered")
        self.assertTrue(processor.exit_requested)

    def test_raising_message_ends_the_loop(self):
        # Unlike a plain timeout, an unexpected exception from wait_for_message
        # is not survived: the loop's except clause breaks rather than continues.
        processor = _FakeProcessor([RuntimeError("transport hiccup"), EXIT])

        module.run_message_loop(processor)

        self.assertEqual(1, processor.calls, "loop must break on the exception, before EXIT")
        self.assertFalse(processor.exit_requested)


if __name__ == "__main__":
    unittest.main()
