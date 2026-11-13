"""Unit tests for the cloud-inference business logic.

External behavior only: an ObjectsPostProcessorMessage carrying `face` objects
and an ImageData block goes in, and the `description` attributes Rekognition
produced are asserted on the way out. The AWS boundary (`aws_utils`) is replaced
wholesale before import -- no live call is ever made, exactly as the v4 test did.
"""

import importlib.util
import os
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock

script_dir = os.path.dirname(os.path.abspath(__file__))
# nxai_postprocessor_sdk lives one level up, in nxai-utilities/
sys.path.insert(0, os.path.join(script_dir, "../nxai-utilities"))

import msgpack
import numpy as np
from PIL import Image

from nxai_postprocessor_sdk import ImageData, ImageOrder, ObjectMetadata, ObjectsPostProcessorMessage

sys.modules["aws_utils"] = MagicMock()

spec = importlib.util.spec_from_file_location(
    "postprocessor_cloud_inference_example",
    os.path.join(script_dir, "postprocessor-cloud-inference-example.py"),
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _make_face(x, y, width, height):
    obj = ObjectMetadata()
    obj.x = x
    obj.y = y
    obj.width = width
    obj.height = height
    return obj


def _hwc_frame(width, height):
    """An HWC RGB frame as the engine sends it: uint8 samples, one per channel.

    The values vary across the frame, so a wrong crop is visible.
    """
    return (np.arange(height * width * 3) % 256).astype(np.uint8).reshape((height, width, 3))


# The plugin correlates a response to its request on DeviceID + "|" + Timestamp,
# so both must be non-falsy in the fixture: the SDK defaults are "" and 0, and a
# fixture that left them there would let the echo assertion pass vacuously.
ECHOED_DEVICE_ID = "cam-echo"
ECHOED_TIMESTAMP = 1726000000000000


def _make_message(faces, frame=None, layout=ImageOrder.HWC, objects=None, device_id=ECHOED_DEVICE_ID, timestamp=ECHOED_TIMESTAMP):
    frame = _hwc_frame(8, 4) if frame is None else frame
    msg = ObjectsPostProcessorMessage()
    msg.device_id = device_id
    msg.timestamp = timestamp
    msg.objects = {"face": faces} if objects is None else objects
    image = ImageData()
    image.height, image.width = frame.shape[0], frame.shape[1]
    image.layout = layout
    if layout == ImageOrder.CHW:
        image.bytes = frame.transpose(2, 0, 1).tobytes()
    else:
        image.bytes = frame.tobytes()
    msg.image_data = image
    return msg


class TestHandleMessage(unittest.TestCase):
    def setUp(self):
        module.classify_faces = MagicMock(return_value="A male in his 30s feeling happy.")
        module.image_path = os.path.join(tempfile.mkdtemp(), "face.png")

    def test_attaches_description_attribute_to_the_face(self):
        msg = _make_message([_make_face(0.25, 0.25, 0.5, 0.5)])

        module.handle_message(msg)

        obj = msg.objects["face"][0]
        self.assertEqual(obj.attributes["description"], "A male in his 30s feeling happy.")

    def test_crops_the_normalized_box_scaled_to_pixels(self):
        frame = _hwc_frame(8, 4)
        msg = _make_message([_make_face(0.25, 0.25, 0.5, 0.5)], frame=frame)

        module.handle_message(msg)

        # The normalized box covers pixels x in [2,6), y in [1,3).
        saved = np.array(Image.open(module.image_path))
        np.testing.assert_array_equal(saved, frame[1:3, 2:6])

    def test_reads_a_chw_frame(self):
        frame = _hwc_frame(8, 4)
        msg = _make_message([_make_face(0.25, 0.25, 0.5, 0.5)], frame=frame, layout=ImageOrder.CHW)

        module.handle_message(msg)

        saved = np.array(Image.open(module.image_path))
        np.testing.assert_array_equal(saved, frame[1:3, 2:6])

    def test_classifies_at_most_two_faces_per_frame(self):
        faces = [_make_face(0.1 * i, 0.1, 0.2, 0.2) for i in range(4)]
        msg = _make_message(faces)

        module.handle_message(msg)

        described = [f for f in msg.objects["face"] if "description" in f.attributes]
        self.assertEqual(len(described), 2)
        self.assertEqual(module.classify_faces.call_count, 2)

    def test_an_undescribed_face_gets_no_attribute_and_does_not_count(self):
        module.classify_faces.side_effect = [None, "first", "second"]
        faces = [_make_face(0.1 * i, 0.1, 0.2, 0.2) for i in range(3)]
        msg = _make_message(faces)

        module.handle_message(msg)

        self.assertNotIn("description", msg.objects["face"][0].attributes)
        self.assertEqual(msg.objects["face"][1].attributes["description"], "first")
        self.assertEqual(msg.objects["face"][2].attributes["description"], "second")

    def test_no_faces_means_no_classification(self):
        msg = _make_message([], objects={"person": [_make_face(0.1, 0.1, 0.2, 0.2)]})

        module.handle_message(msg)

        module.classify_faces.assert_not_called()
        self.assertIn("person", msg.objects)


class TestResponseEchoesCorrelationFields(unittest.TestCase):
    def setUp(self):
        module.classify_faces = MagicMock(return_value="A male in his 30s feeling happy.")
        module.image_path = os.path.join(tempfile.mkdtemp(), "face.png")

    def test_serialized_response_carries_the_requests_device_id_and_timestamp(self):
        # The AI Manager routes a response back to the camera that asked for it by
        # the key DeviceID + "|" + Timestamp. A
        # response that does not echo both fields back unchanged is dropped
        # silently -- no log line anywhere -- so nothing but this assertion
        # catches a processor that answers with a freshly built message.
        # The comparison is against the serialized frame, which is what the
        # plugin actually reads.
        msg = _make_message(
            [_make_face(0.25, 0.25, 0.5, 0.5)], device_id=ECHOED_DEVICE_ID, timestamp=ECHOED_TIMESTAMP
        )

        result = module.handle_message(msg)

        serialized = msgpack.unpackb(result.to_bytes(), raw=False)
        self.assertEqual(ECHOED_DEVICE_ID, serialized["DeviceID"])
        self.assertEqual(ECHOED_TIMESTAMP, serialized["Timestamp"])


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

    def __init__(self, script, send_succeeds=True):
        self._script = list(script)
        self._send_succeeds = send_succeeds
        self.calls = 0
        self.exit_requested = False
        self.responses = []

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
        return _make_message([_make_face(0.1, 0.1, 0.2, 0.2)])

    def write_objects_message(self, msg):
        return b"serialized-response"

    def send_response(self, response_bytes):
        self.responses.append(response_bytes)
        return self._send_succeeds


class TestMessageLoopSurvivability(unittest.TestCase):
    """The engine keeps sending frames; a processor that exits is dropped."""

    def setUp(self):
        module.classify_faces = MagicMock(return_value="A male in his 30s feeling happy.")
        module.image_path = os.path.join(tempfile.mkdtemp(), "face.png")

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
        # The one parseable frame was classified and answered: a loop that skipped it
        # would still reach EXIT with the same call count.
        self.assertEqual([b"serialized-response"], processor.responses)

    def test_unparseable_frame_does_not_end_the_loop(self):
        processor = _FakeProcessor([UNPARSEABLE, b"frame-bytes", EXIT])

        module.run_message_loop(processor)

        self.assertEqual(3, processor.calls, "loop must survive an unparseable frame")
        self.assertEqual([b"serialized-response"], processor.responses)

    def test_an_undescribed_face_does_not_end_the_loop(self):
        # Rekognition returning no description is the unreachable-cloud case: the
        # frame is answered anyway and the loop stays up to see the next EXIT.
        module.classify_faces = MagicMock(return_value=None)
        processor = _FakeProcessor([b"frame-bytes", EXIT])

        module.run_message_loop(processor)

        self.assertEqual(2, processor.calls, "an undescribed face must not end the loop")
        self.assertEqual([b"serialized-response"], processor.responses)

    def test_failed_send_ends_the_loop(self):
        processor = _FakeProcessor([b"frame-bytes", EXIT], send_succeeds=False)

        module.run_message_loop(processor)

        self.assertEqual(1, processor.calls, "loop must break on the failed send, before EXIT")
        self.assertFalse(processor.exit_requested)

    def test_raising_message_ends_the_loop(self):
        # Unlike a plain timeout, an unexpected exception from wait_for_message
        # is not survived: the loop's except clause breaks rather than continues.
        processor = _FakeProcessor([RuntimeError("transport hiccup"), EXIT])

        module.run_message_loop(processor)

        self.assertEqual(1, processor.calls, "loop must break on the exception, before EXIT")
        self.assertFalse(processor.exit_requested)


class TestShutdownLeavesNothingBehind(unittest.TestCase):
    """Leaving the loop has to be the same thing as the process being free to exit.

    The Rekognition call is the only outbound request this processor makes, and the
    concern it raises is a request still in flight when the exit signal arrives. It
    cannot be: the call is a synchronous step of the loop body, so the frame carrying
    it completes before the loop asks for the next message. These tests hold that
    property rather than reading `daemon=True` off a thread that does not exist --
    a statically-read flag would not notice a future non-daemon worker.
    """

    def setUp(self):
        module.image_path = os.path.join(tempfile.mkdtemp(), "face.png")

    def test_the_cloud_request_runs_on_the_loops_own_thread(self):
        classified_on = []
        module.classify_faces = MagicMock(
            side_effect=lambda *args: classified_on.append(threading.current_thread())
            or "A male in his 30s feeling happy."
        )
        processor = _FakeProcessor([b"frame-bytes", EXIT])

        module.run_message_loop(processor)

        self.assertEqual([threading.current_thread()], classified_on)

    def test_no_thread_outlives_the_loop(self):
        module.classify_faces = MagicMock(return_value="A male in his 30s feeling happy.")
        processor = _FakeProcessor([b"frame-bytes", EXIT])
        before = set(threading.enumerate())

        module.run_message_loop(processor)

        self.assertTrue(module.classify_faces.called, "the request path must really have run")
        self.assertEqual(before, set(threading.enumerate()))


if __name__ == "__main__":
    unittest.main()
