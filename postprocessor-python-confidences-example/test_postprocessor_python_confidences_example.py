"""Unit tests for the confidences-example business logic.

External behavior only: a built ObjectsPostProcessorMessage goes in, and the
per-object `Confidence` attribute is asserted on the way out. No IPC, no
sockets.
"""

import importlib.util
import os
import sys
import unittest

import msgpack

script_dir = os.path.dirname(os.path.abspath(__file__))
# nxai_postprocessor_sdk lives one level up, in nxai-utilities/
sys.path.insert(0, os.path.join(script_dir, "../nxai-utilities"))

from nxai_postprocessor_sdk import ObjectMetadata, ObjectsPostProcessorMessage

spec = importlib.util.spec_from_file_location(
    "postprocessor_python_confidences_example",
    os.path.join(script_dir, "postprocessor-python-confidences-example.py"),
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


# The plugin correlates a response to its request on DeviceID + "|" + Timestamp,
# so both must be non-falsy in the fixture: the SDK defaults are "" and 0, and a
# fixture that left them there would let the echo assertion pass vacuously.
ECHOED_DEVICE_ID = "cam-echo"
ECHOED_TIMESTAMP = 1726000000000000


def _make_message(objects, device_id="cam-1", timestamp=ECHOED_TIMESTAMP):
    msg = ObjectsPostProcessorMessage()
    msg.device_id = device_id
    msg.timestamp = timestamp
    msg.objects = objects
    return msg


def _object(confidence):
    obj = ObjectMetadata()
    obj.confidence = confidence
    return obj


class TestHandleMessage(unittest.TestCase):
    def test_adds_confidence_attribute_bare_format(self):
        obj = _object(0.9)
        msg = _make_message({"person": [obj]})
        module.handle_message(msg)
        self.assertEqual(obj.attributes["Confidence"], "0.9")

    def test_rounds_to_two_decimals(self):
        obj = _object(0.5678)
        msg = _make_message({"person": [obj]})
        module.handle_message(msg)
        self.assertEqual(obj.attributes["Confidence"], "0.57")

    def test_no_trailing_zero_padding(self):
        obj = _object(1.0)
        msg = _make_message({"person": [obj]})
        module.handle_message(msg)
        self.assertEqual(obj.attributes["Confidence"], "1.0")

    def test_annotates_every_object_across_all_classes(self):
        person = _object(0.9)
        car = _object(0.42)
        msg = _make_message({"person": [person], "car": [car]})
        module.handle_message(msg)
        self.assertEqual(person.attributes["Confidence"], "0.9")
        self.assertEqual(car.attributes["Confidence"], "0.42")

    def test_no_objects_no_op(self):
        msg = _make_message({})
        result = module.handle_message(msg)
        self.assertEqual(result.objects, {})

    def test_preserves_existing_attributes(self):
        obj = _object(0.9)
        obj.attributes["Existing"] = "value"
        msg = _make_message({"person": [obj]})
        module.handle_message(msg)
        self.assertEqual(obj.attributes["Existing"], "value")
        self.assertEqual(obj.attributes["Confidence"], "0.9")


class TestResponseEchoesCorrelationFields(unittest.TestCase):
    def test_serialized_response_carries_the_requests_device_id_and_timestamp(self):
        # The AI Manager routes a response back to the camera that asked for it by
        # the key DeviceID + "|" + Timestamp. A
        # response that does not echo both fields back unchanged is dropped
        # silently -- no log line anywhere -- so nothing but this assertion
        # catches a processor that answers with a freshly built message.
        # The comparison is against the serialized frame, which is what the
        # plugin actually reads.
        msg = _make_message({"person": [_object(0.9)]}, device_id=ECHOED_DEVICE_ID, timestamp=ECHOED_TIMESTAMP)

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
        return _make_message({"person": [_object(0.9)]})

    def write_objects_message(self, msg):
        # Stand in for msgpack serialization: hand back the annotated objects so the
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
        annotated = processor.sent[0]
        self.assertEqual("0.9", annotated.objects["person"][0].attributes["Confidence"])

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
