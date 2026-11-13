"""Unit tests for the events-example business logic.

External behavior only: a built ObjectsPostProcessorMessage goes in, and the
appended event / description are asserted on the way out. No IPC, no sockets.
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
    "postprocessor_python_events_example",
    os.path.join(script_dir, "postprocessor-python-events-example.py"),
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


def _objects(n):
    return [ObjectMetadata() for _ in range(n)]


class TestFormatDescription(unittest.TestCase):
    def test_no_classes_at_all(self):
        self.assertEqual(
            module._format_description({}),
            "\nThere are no objects in the frame.",
        )

    def test_single_class_with_detections(self):
        self.assertEqual(
            module._format_description({"car": _objects(2)}),
            "\nThere are 2 car's in the frame",
        )

    def test_zero_count_class_before_nonzero_is_discarded(self):
        # v4 quirk, preserved: description_string is fully replaced (not
        # appended to) on the first nonzero-count class, so any zero-count
        # class encountered earlier is silently dropped.
        description = module._format_description(
            {"person": _objects(0), "car": _objects(1)}
        )
        self.assertEqual(description, "\nThere are 1 car's in the frame")

    def test_zero_count_class_after_nonzero_is_appended(self):
        # Same quirk from the other direction: once the "There are" reset has
        # happened, later classes (zero-count or not) append normally.
        description = module._format_description(
            {"car": _objects(1), "person": _objects(0)}
        )
        self.assertEqual(
            description,
            "\nThere are 1 car's in the frame 0 person's in the frame",
        )

    def test_all_classes_zero(self):
        description = module._format_description(
            {"person": _objects(0), "car": _objects(0)}
        )
        self.assertEqual(
            description,
            "\nThere are no objects in the frame. 0 person's in the frame 0 car's in the frame",
        )


class TestHandleMessage(unittest.TestCase):
    def test_event_always_appended_even_with_no_objects(self):
        msg = _make_message({})
        module.handle_message(msg)
        self.assertEqual(len(msg.events), 1)

    def test_event_type_id_is_the_bare_registered_id(self):
        # The DeviceAgent manifest declares
        # external_postprocessors.json's Events[].ID verbatim, so the emitted
        # id is that bare id -- no "nx.nxai." prefix (unlike object classes,
        # which the engine still prefixes).
        msg = _make_message({"car": _objects(1)})
        module.handle_message(msg)
        self.assertEqual(msg.events[0].type_id, "ex.example.event")

    def test_event_description_matches_format_description(self):
        msg = _make_message({"car": _objects(3)})
        module.handle_message(msg)
        self.assertEqual(msg.events[0].description, "\nThere are 3 car's in the frame")

    def test_event_serializes_without_caption(self):
        # v5 drops the v4 Caption: the wire form carries only ID + Description.
        msg = _make_message({"car": _objects(1)})
        module.handle_message(msg)
        self.assertEqual(set(msg.events[0].to_dict().keys()), {"ID", "Description"})


class TestResponseEchoesCorrelationFields(unittest.TestCase):
    def test_serialized_response_carries_the_requests_device_id_and_timestamp(self):
        # The AI Manager routes a response back to the camera that asked for it by
        # the key DeviceID + "|" + Timestamp. A
        # response that does not echo both fields back unchanged is dropped
        # silently -- no log line anywhere -- so nothing but this assertion
        # catches a processor that answers with a freshly built message.
        # The comparison is against the serialized frame, which is what the
        # plugin actually reads.
        msg = _make_message({"car": _objects(1)}, device_id=ECHOED_DEVICE_ID, timestamp=ECHOED_TIMESTAMP)

        result = module.handle_message(msg)

        serialized = msgpack.unpackb(result.to_bytes(), raw=False)
        self.assertEqual(ECHOED_DEVICE_ID, serialized["DeviceID"])
        self.assertEqual(ECHOED_TIMESTAMP, serialized["Timestamp"])


EXIT = object()  # Sentinel script item: simulates IpcSignal.EXIT via exit_requested.
UNPARSEABLE = b"unparseable-frame"  # Script item for which parse_objects_message returns None.
UNSERIALIZABLE = b"unserializable-frame"  # Script item for which write_objects_message returns None.


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
    EXIT" from "kept spinning past it". `sent` records the responses handed to
    send_response, so a frame that was skipped rather than processed is visible.
    """

    def __init__(self, script, send_succeeds=True):
        self._script = list(script)
        self._send_succeeds = send_succeeds
        self.calls = 0
        self.sent = []
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
        msg.objects = {"car": _objects(1)}
        # Carried through so the test can tell which frame produced a response.
        msg.device_id = message_bytes.decode()
        return msg

    def write_objects_message(self, msg):
        if msg.device_id == UNSERIALIZABLE.decode():
            return None
        return msg.device_id.encode()

    def send_response(self, response_bytes):
        self.sent.append(response_bytes)
        return self._send_succeeds


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
        # The one real frame was answered: a loop that skipped it would still
        # reach EXIT with the same call count.
        self.assertEqual([b"frame-bytes"], processor.sent)

    def test_unparseable_frame_does_not_end_the_loop(self):
        # A frame the SDK cannot parse is logged and skipped, not fatal.
        processor = _FakeProcessor([UNPARSEABLE, b"frame-bytes", EXIT])

        module.run_message_loop(processor)

        self.assertEqual(3, processor.calls, "loop must survive an unparseable frame")
        self.assertEqual([b"frame-bytes"], processor.sent)

    def test_unserializable_message_does_not_end_the_loop(self):
        # Serialization failure is skipped the same way, with no response sent.
        processor = _FakeProcessor([UNSERIALIZABLE, b"frame-bytes", EXIT])

        module.run_message_loop(processor)

        self.assertEqual(3, processor.calls, "loop must survive a serialization failure")
        self.assertEqual([b"frame-bytes"], processor.sent)

    def test_failed_send_ends_the_loop(self):
        # A broken return channel is fatal: the loop breaks instead of spinning.
        processor = _FakeProcessor([b"frame-bytes", EXIT], send_succeeds=False)

        module.run_message_loop(processor)

        self.assertEqual(1, processor.calls, "loop must break on a failed send, before EXIT")
        self.assertFalse(processor.exit_requested)

    def test_raising_message_ends_the_loop(self):
        # Unlike a plain timeout, an unexpected exception from wait_for_message
        # is not survived: the loop's except clause breaks rather than continues.
        processor = _FakeProcessor([RuntimeError("transport hiccup"), EXIT])

        module.run_message_loop(processor)

        self.assertEqual(1, processor.calls, "loop must break on the exception, before EXIT")
        self.assertFalse(processor.exit_requested)


if __name__ == "__main__":
    unittest.main()
