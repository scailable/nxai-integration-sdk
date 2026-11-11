"""Unit tests for the object-counter-threshold business logic.

External behavior only: a built ObjectsPostProcessorMessage goes in, and the
per-class counts / total / fired event are asserted on the way out. No IPC, no
sockets. The threshold is read fresh from `msg.user_settings` on every call,
falling back to `DEFAULT_MIN_OBJECT_COUNT` when the key is absent.
"""

import importlib.util
import os
import sys
import unittest

import msgpack

script_dir = os.path.dirname(os.path.abspath(__file__))
# nxai_postprocessor_sdk lives one level up, in nxai-utilities/
sys.path.insert(0, os.path.join(script_dir, "../nxai-utilities"))

import nxai_postprocessor_sdk
from nxai_postprocessor_sdk import ObjectMetadata, ObjectsPostProcessorMessage

spec = importlib.util.spec_from_file_location(
    "postprocessor_python_object_counter_threshold",
    os.path.join(script_dir, "postprocessor-python-object-counter-threshold.py"),
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


# The plugin correlates a response to its request on DeviceID + "|" + Timestamp,
# so both must be non-falsy in the fixture: the SDK defaults are "" and 0, and a
# fixture that left them there would let the echo assertion pass vacuously.
ECHOED_DEVICE_ID = "cam-echo"
ECHOED_TIMESTAMP = 1726000000000000


def _make_message(objects, user_settings=None, device_id="cam-1", timestamp=ECHOED_TIMESTAMP):
    msg = ObjectsPostProcessorMessage()
    msg.objects = objects
    msg.user_settings = user_settings or {}
    msg.device_id = device_id
    msg.timestamp = timestamp
    return msg


def _objects(n):
    return [ObjectMetadata() for _ in range(n)]


class TestCountObjects(unittest.TestCase):
    def test_count_per_class(self):
        msg = _make_message({"person": _objects(2), "car": _objects(1)})
        self.assertEqual(module.count_objects(msg), {"person": 2, "car": 1})

    def test_empty_message_counts_nothing(self):
        self.assertEqual(module.count_objects(_make_message({})), {})


class TestThresholdEvent(unittest.TestCase):
    def _fired(self, msg):
        # The emitted id is the bare id from the JSON: the AI Manager copies
        # Events[].ID into the manifest verbatim and relays emitted ids
        # verbatim, so only the bare id survives the mediaserver's check.
        return any(
            e.type_id == "object.counter.threshold" for e in msg.events
        )

    def test_just_below_default_threshold_no_event(self):
        msg = _make_message({"person": _objects(4)})
        module.handle_message(msg)
        self.assertFalse(self._fired(msg))

    def test_exactly_at_default_threshold_fires(self):
        msg = _make_message({"person": _objects(5)})
        module.handle_message(msg)
        self.assertTrue(self._fired(msg))

    def test_above_default_threshold_fires(self):
        msg = _make_message({"person": _objects(3), "car": _objects(3)})
        module.handle_message(msg)
        self.assertTrue(self._fired(msg))

    def test_live_setting_overrides_default(self):
        # 3 objects would not fire against the hardcoded default (5); the live
        # setting lowers the threshold below the seeded count.
        msg = _make_message(
            {"person": _objects(3)}, user_settings={"min_object_count": "3"}
        )
        module.handle_message(msg)
        self.assertTrue(self._fired(msg))

    def test_absent_setting_key_falls_back_to_default(self):
        msg = _make_message({"person": _objects(5)}, user_settings={})
        module.handle_message(msg)
        self.assertTrue(self._fired(msg))

    def test_event_serializes_without_caption(self):
        msg = _make_message({"person": _objects(5)})
        module.handle_message(msg)
        event = next(e for e in msg.events if e.type_id == "object.counter.threshold")
        # v5 drops the v4 Caption: the wire form carries only ID + Description.
        self.assertEqual(set(event.to_dict().keys()), {"ID", "Description"})

    def test_no_objects_no_event(self):
        msg = _make_message({})
        module.handle_message(msg)
        self.assertEqual(msg.events, [])


class TestResponseEchoesCorrelationFields(unittest.TestCase):
    def test_serialized_response_carries_the_requests_device_id_and_timestamp(self):
        # The AI Manager routes a response back to the camera that asked for it by
        # the key DeviceID + "|" + Timestamp. A
        # response that does not echo both fields back unchanged is dropped
        # silently -- no log line anywhere -- so nothing but this assertion
        # catches a processor that answers with a freshly built message.
        # The comparison is against the serialized frame, which is what the
        # plugin actually reads.
        msg = _make_message(
            {"person": _objects(5)}, device_id=ECHOED_DEVICE_ID, timestamp=ECHOED_TIMESTAMP
        )

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
    EXIT" from "kept spinning past it". `sent` records every response handed to
    send_response(), so a frame that was skipped rather than processed is visible.
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
        return item

    def parse_objects_message(self, message_bytes):
        # Every scripted frame parses: five people, enough to fire the event.
        return _make_message({"person": _objects(5)})

    def write_objects_message(self, msg):
        return b"serialized:" + str(len(msg.events)).encode()

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
        self.assertEqual([b"serialized:1"], processor.sent)



if __name__ == "__main__":
    unittest.main()
