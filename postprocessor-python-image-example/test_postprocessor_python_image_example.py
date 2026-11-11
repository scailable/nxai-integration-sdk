"""Unit tests for the image-example business logic.

External behavior only: a built ObjectsPostProcessorMessage with an attached
ImageData goes in, and the injected "python_image_example" object plus its
ImageDecodeOk/ImageDecodeError attributes are asserted on the way out. No IPC,
no sockets, no HTTP server -- `_publish_frame` just updates module globals.

Every frame here is built from uint8 values, one byte per channel, matching
what this example's `_decode_frame` expects. A fixture built at the wrong
element size would let a change that assumes the wrong size pass here and
return garbage on a live camera.
"""

import importlib.util
import os
import sys
import unittest

import msgpack

script_dir = os.path.dirname(os.path.abspath(__file__))
# nxai_postprocessor_sdk lives one level up, in nxai-utilities/
sys.path.insert(0, os.path.join(script_dir, "../nxai-utilities"))

from nxai_postprocessor_sdk import ImageData, ObjectsPostProcessorMessage

spec = importlib.util.spec_from_file_location(
    "postprocessor_python_image_example",
    os.path.join(script_dir, "postprocessor-python-image-example.py"),
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _frame(values):
    """A frame as the engine sends it: one uint8 byte per channel."""
    return bytes(values)


# A valid 1x2 RGB frame, HWC (ImageData's default layout): six uint8 bytes,
# matching height=1 x width=2 x 3 channels.
FRAME_HEIGHT, FRAME_WIDTH = 1, 2
FRAME = _frame([1, 2, 3, 4, 5, 6])

# One byte short of FRAME_HEIGHT x FRAME_WIDTH x 3 -- a truncated frame a
# plain byte sum would accept without complaint, since it never checks shape.
CORRUPT_FRAME = _frame([1, 2, 3, 4, 5])

# The plugin correlates a response to its request on DeviceID + "|" + Timestamp,
# so both must be non-falsy in the fixture: the SDK defaults are "" and 0, and a
# fixture that left them there would let the echo assertion pass vacuously.
ECHOED_DEVICE_ID = "cam-echo"
ECHOED_TIMESTAMP = 1726000000000000


def _make_message(
    image_bytes,
    objects=None,
    device_id="cam-1",
    timestamp=ECHOED_TIMESTAMP,
    height=FRAME_HEIGHT,
    width=FRAME_WIDTH,
):
    msg = ObjectsPostProcessorMessage()
    msg.device_id = device_id
    msg.timestamp = timestamp
    msg.objects = objects if objects is not None else {}
    image = ImageData()
    image.bytes = image_bytes
    image.height = height
    image.width = width
    msg.image_data = image
    return msg


class TestHandleMessage(unittest.TestCase):
    def test_injects_test_object_with_normalized_box(self):
        msg = _make_message(FRAME)

        module.handle_message(msg)

        self.assertIn("python_image_example", msg.objects)
        self.assertEqual(len(msg.objects["python_image_example"]), 1)
        obj = msg.objects["python_image_example"][0]
        self.assertEqual(obj.x, 0.1)
        self.assertEqual(obj.y, 0.1)
        self.assertEqual(obj.width, 0.1)
        self.assertEqual(obj.height, 0.1)
        self.assertEqual(obj.confidence, 1.0)

    def test_first_call_creates_track_second_call_continues_it(self):
        module.test_track_ids = {}
        first_msg = _make_message(FRAME)
        second_msg = _make_message(FRAME)

        first_result = module.handle_message(first_msg)
        second_result = module.handle_message(second_msg)

        first_obj = first_result.objects["python_image_example"][0]
        self.assertIsNotNone(first_obj.new_track_id)
        self.assertNotEqual(first_obj.new_track_id, "")
        self.assertIsNone(first_obj.track_id)

        second_obj = second_result.objects["python_image_example"][0]
        self.assertIsNone(second_obj.new_track_id)
        self.assertEqual(second_obj.track_id, first_obj.new_track_id)

    def test_each_camera_gets_its_own_track(self):
        # A track UUID belongs to the device that minted it: the plugin looks an
        # incoming ID up in that device's own track table, so continuing camera
        # one's track on camera two is rejected with "Could not find object ID"
        # and the object is dropped. Enabling this example on a second camera
        # must mint a NewID there, then continue each camera's own track.
        module.test_track_ids = {}

        first = [
            module.handle_message(_make_message(FRAME, device_id="cam-1")),
            module.handle_message(_make_message(FRAME, device_id="cam-2")),
        ]
        second = [
            module.handle_message(_make_message(FRAME, device_id="cam-1")),
            module.handle_message(_make_message(FRAME, device_id="cam-2")),
        ]

        new_ids = [r.objects["python_image_example"][0].new_track_id for r in first]
        self.assertTrue(all(new_ids))
        self.assertNotEqual(new_ids[0], new_ids[1])
        for new_id, result in zip(new_ids, second):
            obj = result.objects["python_image_example"][0]
            self.assertIsNone(obj.new_track_id)
            self.assertEqual(obj.track_id, new_id)

    def test_sets_image_decode_ok_for_a_well_formed_frame(self):
        msg = _make_message(FRAME)

        module.handle_message(msg)

        obj = msg.objects["python_image_example"][0]
        self.assertEqual(obj.attributes["ImageDecodeOk"], "True")
        self.assertNotIn("ImageDecodeError", obj.attributes)

    def test_flags_a_truncated_frame_that_a_byte_sum_would_miss(self):
        # CORRUPT_FRAME has a valid-looking byte sum but one byte short of
        # what height x width x channels requires -- exactly the kind of
        # corruption the old byte-sum check could never catch.
        msg = _make_message(CORRUPT_FRAME)

        module.handle_message(msg)

        obj = msg.objects["python_image_example"][0]
        self.assertEqual(obj.attributes["ImageDecodeOk"], "False")
        self.assertIn("ImageDecodeError", obj.attributes)

    def test_raises_on_a_missing_image_data_block(self):
        # A missing ImageData block (ReceiveInputImage declared but no block arrived)
        # is a different failure than a malformed-but-present frame: it's left to
        # propagate rather than caught, matching
        # tests/e2e/CMakeLists.txt's e2e-python-negative-no-image pin. Catching it here
        # too would quietly change that contract.
        msg = _make_message(FRAME)
        msg.image_data = None

        with self.assertRaises(AttributeError):
            module.handle_message(msg)

    def test_publishes_the_decoded_frame_for_the_preview_stream(self):
        module._latest_frame_jpeg = None
        msg = _make_message(FRAME)

        module.handle_message(msg)

        self.assertIsNotNone(module._latest_frame_jpeg)

    def test_does_not_publish_on_a_corrupt_frame(self):
        module._latest_frame_jpeg = None
        msg = _make_message(CORRUPT_FRAME)

        module.handle_message(msg)

        self.assertIsNone(module._latest_frame_jpeg)

    def test_preserves_existing_classes(self):
        msg = _make_message(FRAME, objects={"person": []})

        module.handle_message(msg)

        self.assertIn("person", msg.objects)
        self.assertIn("python_image_example", msg.objects)


class TestResponseEchoesCorrelationFields(unittest.TestCase):
    def test_serialized_response_carries_the_requests_device_id_and_timestamp(self):
        # The AI Manager routes a response back to the camera that asked for it by
        # the key DeviceID + "|" + Timestamp. A
        # response that does not echo both fields back unchanged is dropped
        # silently -- no log line anywhere -- so nothing but this assertion
        # catches a processor that answers with a freshly built message.
        # The comparison is against the serialized frame, which is what the
        # plugin actually reads.
        msg = _make_message(FRAME, device_id=ECHOED_DEVICE_ID, timestamp=ECHOED_TIMESTAMP)

        result = module.handle_message(msg)

        serialized = msgpack.unpackb(result.to_bytes(), raw=False)
        self.assertEqual(ECHOED_DEVICE_ID, serialized["DeviceID"])
        self.assertEqual(ECHOED_TIMESTAMP, serialized["Timestamp"])


EXIT = object()  # Sentinel script item: simulates IpcSignal.EXIT via exit_requested.


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
    EXIT" from "kept spinning past it". `responded` records the messages handed to
    write_objects_message, so a frame that was skipped rather than processed shows up.
    """

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0
        self.exit_requested = False
        self.responded = []

    def wait_for_message(self, timeout_ms=None):
        if not self._script:
            raise _ScriptExhausted("wait_for_message called past the end of the script")
        self.calls += 1
        item = self._script.pop(0)
        if item is EXIT:
            self.exit_requested = True
            return None
        return item

    def parse_objects_message(self, message_bytes):
        # This example reads the input image block, so a scripted frame must carry
        # ImageData or handle_message cannot sum its bytes.
        return _make_message(FRAME, device_id="device-from-script")

    def write_objects_message(self, msg):
        self.responded.append(msg)
        return b"response-bytes"

    def send_response(self, response_bytes):
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
        self.assertEqual(1, len(processor.responded))
        self.assertIn("python_image_example", processor.responded[0].objects)


if __name__ == "__main__":
    unittest.main()
