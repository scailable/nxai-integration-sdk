"""Unit tests for the Edge Impulse uploader's business logic.

External behavior only: an ObjectsPostProcessorMessage carrying detections and
an ImageData block goes in, and what ends up queued for (or sent to) Edge
Impulse is asserted on the way out. The Edge Impulse boundary (`edgeimpulse`) is
replaced wholesale before import -- no live call is ever made, exactly as the v4
test did.
"""

import importlib.util
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock

script_dir = os.path.dirname(os.path.abspath(__file__))
# nxai_postprocessor_sdk lives one level up, in nxai-utilities/
sys.path.insert(0, os.path.join(script_dir, "../nxai-utilities"))

import numpy as np

from nxai_postprocessor_sdk import ImageData, ImageOrder, ObjectMetadata, ObjectsPostProcessorMessage

sys.modules["edgeimpulse"] = MagicMock()

spec = importlib.util.spec_from_file_location(
    "postprocessor_python_edgeimpulse_example",
    os.path.join(script_dir, "postprocessor-python-edgeimpulse-example.py"),
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

JPEG_MAGIC = b"\xff\xd8"


def _frame(width=8, height=4):
    """An HWC RGB frame whose bytes vary, so a wrong conversion is visible."""
    return np.arange(height * width * 3, dtype=np.uint8).reshape((height, width, 3))


def _make_message(confidences, frame=None, layout=ImageOrder.HWC):
    frame = _frame() if frame is None else frame
    msg = ObjectsPostProcessorMessage()
    msg.objects = {"car": [_detection(confidence) for confidence in confidences]}
    image = ImageData()
    image.height, image.width = frame.shape[0], frame.shape[1]
    image.layout = layout
    pixels = frame
    if layout == ImageOrder.CHW:
        pixels = pixels.transpose(2, 0, 1)
    image.bytes = pixels.tobytes()
    msg.image_data = image
    return msg


def _detection(confidence):
    obj = ObjectMetadata()
    obj.confidence = confidence
    return obj


class TestThresholdUpload(unittest.TestCase):
    def setUp(self):
        module.samples_buffer = []
        module.auto_generator = False
        module.p_value = 0.4
        # High enough that no test here trips a flush; batching is TestBatchFlush's.
        module.samples_buffer_flush_size = 20

    def test_queues_the_frame_when_a_detection_is_below_p_value(self):
        """The uploader's whole point: an uncertain detection makes its frame a
        training sample, queued as JPEG bytes."""
        module.handle_message(_make_message([0.3]))

        self.assertEqual(len(module.samples_buffer), 1)
        self.assertTrue(module.samples_buffer[0].startswith(JPEG_MAGIC))

    def test_skips_the_frame_when_every_detection_is_confident(self):
        """A confident frame teaches Edge Impulse nothing, so it is not uploaded."""
        module.handle_message(_make_message([0.4, 0.9]))

        self.assertEqual(module.samples_buffer, [])

    def test_logs_the_queue_depth_after_the_frame_became_a_jpeg(self):
        """An end-to-end test's verdict is this line: it is emitted only
        after the frame conversion produced JPEG bytes, so a passing
        log check proves the image path ran, not merely that a frame was
        considered."""
        with self.assertLogs(module.logger, level="INFO") as captured:
            module.handle_message(_make_message([0.3]))

        self.assertIn("Queued sample: 1 pending upload", captured.output[-1])

    def test_encodes_a_uint8_chw_frame_like_its_uint8_hwc_equivalent(self):
        """A CHW frame and an HWC frame of the same picture must reach Edge Impulse
        as the same picture."""
        frame = _frame()
        module.handle_message(_make_message([0.3], frame=frame, layout=ImageOrder.HWC))
        module.handle_message(_make_message([0.3], frame=frame, layout=ImageOrder.CHW))

        self.assertEqual(module.samples_buffer[1], module.samples_buffer[0])


class TestAutoGenerator(unittest.TestCase):
    def setUp(self):
        module.samples_buffer = []
        module.samples_buffer_flush_size = 20
        module.auto_generator = True
        module.auto_generator_every_seconds = 1
        # Everything below p_value would be uploaded on the threshold path, so a
        # p_value of 0 proves the timer path ignores the score entirely.
        module.p_value = 0.0

    def test_queues_a_confident_frame_once_the_interval_has_elapsed(self):
        """On the timer path the score is irrelevant: the sample is due, so it goes."""
        module.auto_generator_last_sample_at = time.time() - 10

        module.handle_message(_make_message([0.9]))

        self.assertEqual(len(module.samples_buffer), 1)

    def test_does_not_queue_again_until_the_interval_has_elapsed_once_more(self):
        """The interval has to be re-armed after a sample, or every frame of the
        stream is uploaded from then on."""
        module.auto_generator_last_sample_at = time.time() - 10

        module.handle_message(_make_message([0.9]))
        module.handle_message(_make_message([0.9]))

        self.assertEqual(len(module.samples_buffer), 1)


class TestBatchFlush(unittest.TestCase):
    def setUp(self):
        module.samples_buffer = []
        module.samples_counter = 0
        module.auto_generator = False
        module.p_value = 0.4
        module.samples_buffer_flush_size = 2
        module.edgeimpulse.reset_mock()

    def test_uploads_and_drains_the_buffer_once_it_reaches_the_flush_size(self):
        """Samples are batched: nothing leaves until the buffer is full, and then
        the whole batch goes at once and the buffer starts over."""
        module.handle_message(_make_message([0.3]))
        self.assertEqual(module.edgeimpulse.experimental.data.upload_samples.call_count, 0)

        module.handle_message(_make_message([0.3]))

        uploaded = module.edgeimpulse.experimental.data.upload_samples.call_args.args[0]
        self.assertEqual(len(uploaded), 2)
        self.assertEqual(module.samples_buffer, [])

    def test_gives_every_uploaded_sample_a_unique_filename(self):
        """Two frames a second apart would otherwise collide on the timestamp and
        overwrite each other in the project."""
        module.samples_buffer_flush_size = 1
        module.handle_message(_make_message([0.3]))
        module.handle_message(_make_message([0.3]))

        filenames = [
            call.kwargs["filename"]
            for call in module.edgeimpulse.experimental.data.Sample.call_args_list
        ]
        self.assertEqual(len(set(filenames)), 2)


class TestConfigFile(unittest.TestCase):
    """The ini file is this processor's whole settings surface (it exposes no
    NX-plugin UserSettings), so what it parses out of it is external behavior."""

    def _write_config(self, body):
        path = os.path.join(self.tmp.name, "plugin.edgeimpulse.ini")
        with open(path, "w") as handle:
            handle.write(body)
        return path

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_reads_the_settings_out_of_the_ini_file(self):
        path = self._write_config(
            "[edgeimpulse]\n"
            "api_key = ei_live_key\n"
            "auto_generator = True\n"
            "auto_generator_every_seconds = 7\n"
            "samples_buffer_flush_size = 3\n"
            "p_value = 0.75\n"
        )

        module.read_config(path)

        self.assertEqual(module.edge_impulse_api_key, "ei_live_key")
        self.assertIs(module.auto_generator, True)
        self.assertEqual(module.auto_generator_every_seconds, 7)
        self.assertEqual(module.samples_buffer_flush_size, 3)
        self.assertEqual(module.p_value, 0.75)

    def test_treats_a_false_auto_generator_as_off(self):
        """`auto_generator = False` is what the README tells users to write for the
        threshold path; read as a bare string it would be truthy and silently keep
        the timer running."""
        path = self._write_config("[edgeimpulse]\nauto_generator = False\n")

        module.read_config(path)

        self.assertIs(module.auto_generator, False)

    def test_rejects_the_key_of_an_unedited_copy_of_the_shipped_example(self):
        """The README tells users to start from the example file, so a copy nobody
        edited has to stop the processor at startup, not at the first upload."""
        module.read_config(os.path.join(script_dir, "plugin.edgeimpulse.ini.example"))

        self.assertFalse(module.api_key_is_set())

    def test_accepts_a_configured_key(self):
        module.read_config(self._write_config("[edgeimpulse]\napi_key = ei_live_key\n"))

        self.assertTrue(module.api_key_is_set())


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
    EXIT" from "kept spinning past it". This processor is fire-and-forget -- the loop
    calls only handle_message(), never write_objects_message/send_response -- so this
    stub implements only wait_for_message and parse_objects_message.
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
        return _make_message([0.3])


class TestMessageLoopSurvivability(unittest.TestCase):
    """The engine keeps sending frames; a processor that exits is dropped."""

    def setUp(self):
        module.samples_buffer = []
        module.auto_generator = False
        module.p_value = 0.4
        module.samples_buffer_flush_size = 20

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
        self.assertEqual(1, len(module.samples_buffer))

    def test_unparseable_frame_does_not_end_the_loop(self):
        # A frame the SDK cannot parse is logged and skipped, not fatal.
        processor = _FakeProcessor([UNPARSEABLE, b"frame-bytes", EXIT])

        module.run_message_loop(processor)

        self.assertEqual(3, processor.calls, "loop must survive an unparseable frame")
        self.assertEqual(1, len(module.samples_buffer), "only the parseable frame is handled")
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
