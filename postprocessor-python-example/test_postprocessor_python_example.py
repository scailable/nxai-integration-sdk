"""Unit tests for the python-example business logic.

External behavior only: a built ObjectsPostProcessorMessage goes in, and the
injected test object is inspected on the way out. No IPC, no sockets. Mirrors
the construction style used in nxai-utilities/tests/test_message_roundtrip.py.

TestMessageLoopSurvivability below covers the EXIT-vs-timeout distinction:
wait_for_message() returns None both on a plain timeout and on an IPC EXIT
signal, and only exit_requested tells them apart. The loop lives in
run_message_loop() in postprocessor-python-example.py, so a scripted stub
channel can drive it directly.
"""

import os
import sys
import unittest

import msgpack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "nxai-utilities"))

from nxai_postprocessor_sdk import ObjectsPostProcessorMessage, load_config

from importlib import import_module

postprocessor_example = import_module("postprocessor-python-example")


# The plugin correlates a response to its request on DeviceID + "|" + Timestamp,
# so both must be non-falsy in the fixture: the SDK defaults are "" and 0, and a
# fixture that left them there would let the echo assertion pass vacuously.
ECHOED_DEVICE_ID = "cam-echo"
ECHOED_TIMESTAMP = 1726000000000000


def _make_message(objects=None, device_id="cam-1", timestamp=ECHOED_TIMESTAMP):
    msg = ObjectsPostProcessorMessage()
    msg.device_id = device_id
    msg.timestamp = timestamp
    if objects is not None:
        msg.objects = objects
    return msg


class HandleMessageTestCase(unittest.TestCase):
    """Base for the handle_message tests.

    The injected object's track UUIDs are a module global that persists across
    calls by design, so every test starts from a fresh set of tracks.
    """

    def setUp(self):
        postprocessor_example.test_track_ids = {}


class TestInjectedObjectBoundingBox(HandleMessageTestCase):
    def test_box_stays_within_unit_square(self):
        result = postprocessor_example.handle_message(_make_message())

        test_obj = result.objects[postprocessor_example.INJECTED_CLASS][0]
        self.assertGreaterEqual(test_obj.x, 0.0)
        self.assertGreaterEqual(test_obj.y, 0.0)
        self.assertLessEqual(test_obj.x + test_obj.width, 1.0)
        self.assertLessEqual(test_obj.y + test_obj.height, 1.0)


class TestInjectedClassCoexistsWithExistingObjects(HandleMessageTestCase):
    def test_empty_objects_dict_gets_injected_class(self):
        result = postprocessor_example.handle_message(_make_message({}))

        self.assertEqual(len(result.objects[postprocessor_example.INJECTED_CLASS]), 1)

    def test_injected_class_already_present_with_empty_list_appends(self):
        msg = _make_message({postprocessor_example.INJECTED_CLASS: []})

        result = postprocessor_example.handle_message(msg)

        self.assertEqual(len(result.objects[postprocessor_example.INJECTED_CLASS]), 1)

    def test_unknown_class_is_left_untouched(self):
        result = postprocessor_example.handle_message(_make_message({"unrelated_class": []}))

        self.assertIn("unrelated_class", result.objects)
        self.assertEqual(result.objects["unrelated_class"], [])
        self.assertEqual(len(result.objects[postprocessor_example.INJECTED_CLASS]), 1)


class TestInjectedObjectRoundTrip(HandleMessageTestCase):
    def test_attributes_and_confidence_survive_serialization(self):
        result = postprocessor_example.handle_message(_make_message())
        parsed = ObjectsPostProcessorMessage.from_bytes(result.to_bytes())

        test_obj = parsed.objects[postprocessor_example.INJECTED_CLASS][0]
        self.assertEqual(test_obj.confidence, 0.95)
        self.assertEqual(test_obj.attributes["source"], "postprocessor-example")


class TestInjectedObjectTrackContinuity(HandleMessageTestCase):
    def test_second_call_continues_first_calls_track(self):
        first_result = postprocessor_example.handle_message(_make_message())
        second_result = postprocessor_example.handle_message(_make_message())

        first_test_obj = first_result.objects[postprocessor_example.INJECTED_CLASS][0]
        self.assertIsNotNone(first_test_obj.new_track_id)
        self.assertNotEqual(first_test_obj.new_track_id, "")
        self.assertIsNone(first_test_obj.track_id)

        second_test_obj = second_result.objects[postprocessor_example.INJECTED_CLASS][0]
        self.assertIsNone(second_test_obj.new_track_id)
        self.assertEqual(second_test_obj.track_id, first_test_obj.new_track_id)

    def test_a_second_camera_mints_its_own_track(self):
        # A track UUID belongs to the device that minted it: the plugin looks an
        # incoming ID up in that device's own track table, so continuing camera
        # one's track on camera two is rejected with "Could not find object ID"
        # and the object is dropped. Each camera must see a NewID of its own.
        first_camera = postprocessor_example.handle_message(_make_message(device_id="cam-1"))
        second_camera = postprocessor_example.handle_message(_make_message(device_id="cam-2"))

        first_obj = first_camera.objects[postprocessor_example.INJECTED_CLASS][0]
        second_obj = second_camera.objects[postprocessor_example.INJECTED_CLASS][0]
        self.assertIsNotNone(second_obj.new_track_id)
        self.assertIsNone(second_obj.track_id)
        self.assertNotEqual(second_obj.new_track_id, first_obj.new_track_id)

    def test_each_camera_continues_its_own_track(self):
        first_frames = [
            postprocessor_example.handle_message(_make_message(device_id="cam-1")),
            postprocessor_example.handle_message(_make_message(device_id="cam-2")),
        ]
        second_frames = [
            postprocessor_example.handle_message(_make_message(device_id="cam-1")),
            postprocessor_example.handle_message(_make_message(device_id="cam-2")),
        ]

        for first, second in zip(first_frames, second_frames):
            first_obj = first.objects[postprocessor_example.INJECTED_CLASS][0]
            second_obj = second.objects[postprocessor_example.INJECTED_CLASS][0]
            self.assertEqual(second_obj.track_id, first_obj.new_track_id)


class TestResponseEchoesCorrelationFields(HandleMessageTestCase):
    def test_serialized_response_carries_the_requests_device_id_and_timestamp(self):
        # The AI Manager routes a response back to the camera that asked for it by
        # the key DeviceID + "|" + Timestamp. A
        # response that does not echo both fields back unchanged is dropped
        # silently -- no log line anywhere -- so nothing but this assertion
        # catches a processor that answers with a freshly built message.
        # The comparison is against the serialized frame, which is what the
        # plugin actually reads.
        msg = _make_message(device_id=ECHOED_DEVICE_ID, timestamp=ECHOED_TIMESTAMP)

        result = postprocessor_example.handle_message(msg)

        serialized = msgpack.unpackb(result.to_bytes(), raw=False)
        self.assertEqual(ECHOED_DEVICE_ID, serialized["DeviceID"])
        self.assertEqual(ECHOED_TIMESTAMP, serialized["Timestamp"])


class TestProcessorNameFromConfig(unittest.TestCase):
    def test_processor_name_taken_from_flag_not_position(self):
        # The AI Manager launches the processor as: <script> <input_sync>
        # <output_sync> [--Config-key value ...] --ProcessorName <name>, and the
        # example hands that whole sys.argv to load_config(). Since load_config()
        # only keys on "--flag value" pairs, the two leading IPC sync descriptors
        # never become config keys and the name comes from the flag.
        argv = [
            "postprocessor-python-example.py",
            "input-sync-key",
            "output-sync-key",
            "--ProcessorName",
            "python-example",
        ]

        config = load_config(argv)

        self.assertEqual(config.get("ProcessorName", postprocessor_example.PROCESSOR_NAME), "python-example")
        self.assertNotIn("input-sync-key", config)
        self.assertNotIn("output-sync-key", config)

    def test_falls_back_to_default_name_when_flag_absent(self):
        argv = ["postprocessor-python-example.py", "input-sync-key", "output-sync-key"]

        config = load_config(argv)

        self.assertEqual(
            config.get("ProcessorName", postprocessor_example.PROCESSOR_NAME),
            postprocessor_example.PROCESSOR_NAME,
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
    EXIT" from "kept spinning past it". `sent` collects the serialized responses, so a
    frame that was received but never annotated and returned is visible too.
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

    def parse_objects_message(self, message_bytes):
        if message_bytes == UNPARSEABLE:
            return None
        return _make_message({})

    def write_objects_message(self, msg):
        # Stand in for msgpack serialization: hand back the annotated objects so the
        # test can assert the frame really went through handle_message.
        return msg

    def send_response(self, response_bytes):
        self.sent.append(response_bytes)
        return True


class TestMessageLoopSurvivability(unittest.TestCase):
    """The engine keeps sending frames; a processor that exits is dropped."""

    def setUp(self):
        postprocessor_example.test_track_ids = {}

    def test_exit_signal_ends_the_loop_cleanly(self):
        # A graceful EXIT must return, not be swallowed as a plain timeout that
        # keeps the loop spinning until the AI Manager kills the process. The call
        # count is the assertion that catches the swallow: a loop that treats EXIT
        # as a timeout asks for a second message.
        processor = _FakeProcessor([EXIT])

        with self.assertLogs(postprocessor_example.logger, level="INFO") as captured:
            postprocessor_example.run_message_loop(processor)

        self.assertEqual(1, processor.calls, "loop must return on EXIT, not ask for more")
        self.assertTrue(
            any("graceful exit signal" in record.getMessage() for record in captured.records)
        )

    def test_timeout_does_not_end_the_loop(self):
        processor = _FakeProcessor([None, None, b"frame-bytes", EXIT])

        postprocessor_example.run_message_loop(processor)

        self.assertEqual(4, processor.calls, "loop must survive both timeouts and the frame")
        self.assertTrue(processor.exit_requested)
        # The one frame was actually handled: a loop that skipped it would still
        # reach EXIT with the same call count.
        self.assertEqual(1, len(processor.sent))
        annotated = processor.sent[0]
        self.assertEqual(
            1, len(annotated.objects[postprocessor_example.INJECTED_CLASS])
        )

    def test_unparseable_frame_does_not_end_the_loop(self):
        # A frame the SDK cannot parse is logged and skipped, not fatal.
        processor = _FakeProcessor([UNPARSEABLE, b"frame-bytes", EXIT])

        postprocessor_example.run_message_loop(processor)

        self.assertEqual(3, processor.calls, "loop must survive an unparseable frame")
        self.assertEqual(1, len(processor.sent), "only the parseable frame is answered")
        self.assertTrue(processor.exit_requested)

    def test_raising_message_ends_the_loop(self):
        # Unlike a plain timeout, an unexpected exception from wait_for_message
        # is not survived: the loop's except clause breaks rather than continues.
        processor = _FakeProcessor([RuntimeError("transport hiccup"), EXIT])

        postprocessor_example.run_message_loop(processor)

        self.assertEqual(1, processor.calls, "loop must break on the exception, before EXIT")
        self.assertFalse(processor.exit_requested)


if __name__ == "__main__":
    unittest.main()
