"""Unit tests for the settings example's business logic (`handle_message`).

External behavior only: a fake ObjectsPostProcessorMessage carrying forwarded
`user_settings` plus objects goes in, and the forwarded attribute is asserted on
every object that comes out. No IPC, no sockets, no private helpers.

TestRegistrationJson additionally guards the declaration side: the colocated
external_postprocessors.json must declare its settings model under the v5
`UserSettings` key (not the stale `Settings`), the key the engine forwards from.
"""

import importlib.util
import json
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
    "postprocessor_python_settings_example",
    os.path.join(script_dir, "postprocessor-python-settings-example.py"),
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
handle_message = module.handle_message


# The plugin correlates a response to its request on DeviceID + "|" + Timestamp,
# so both must be non-falsy in the fixture: the SDK defaults are "" and 0, and a
# fixture that left them there would let the echo assertion pass vacuously.
ECHOED_DEVICE_ID = "cam-echo"
ECHOED_TIMESTAMP = 1726000000000000


def _make_message(user_settings, objects, device_id="cam-1", timestamp=ECHOED_TIMESTAMP):
    msg = ObjectsPostProcessorMessage()
    msg.user_settings = user_settings
    msg.objects = objects
    msg.device_id = device_id
    msg.timestamp = timestamp
    return msg


class TestHandleMessage(unittest.TestCase):
    def test_forwarded_attribute_applied_to_every_object_in_every_class(self):
        person_a, person_b, car = ObjectMetadata(), ObjectMetadata(), ObjectMetadata()
        msg = _make_message(
            {"attributeName": "reviewed", "attributeValue": "true"},
            {"person": [person_a, person_b], "car": [car]},
        )

        handle_message(msg)

        for obj in (person_a, person_b, car):
            self.assertEqual(obj.attributes["reviewed"], "true")

    def test_attribute_name_and_value_come_from_user_settings(self):
        obj = ObjectMetadata()
        msg = _make_message(
            {"attributeName": "zone", "attributeValue": "loading-dock"},
            {"person": [obj]},
        )

        handle_message(msg)

        self.assertEqual(obj.attributes, {"zone": "loading-dock"})

    def test_no_op_when_user_settings_empty(self):
        obj = ObjectMetadata()
        msg = _make_message({}, {"person": [obj]})

        handle_message(msg)

        self.assertEqual(obj.attributes, {})

    def test_no_op_when_only_one_key_present(self):
        obj = ObjectMetadata()
        msg = _make_message({"attributeName": "reviewed"}, {"person": [obj]})

        handle_message(msg)

        self.assertEqual(obj.attributes, {})


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
            {"attributeName": "reviewed", "attributeValue": "true"},
            {"person": [ObjectMetadata()]},
            device_id=ECHOED_DEVICE_ID,
            timestamp=ECHOED_TIMESTAMP,
        )

        result = handle_message(msg)

        serialized = msgpack.unpackb(result.to_bytes(), raw=False)
        self.assertEqual(ECHOED_DEVICE_ID, serialized["DeviceID"])
        self.assertEqual(ECHOED_TIMESTAMP, serialized["Timestamp"])


class TestRegistrationJson(unittest.TestCase):
    """The colocated registration entry must declare its settings model under the
    v5 `UserSettings` key; the engine forwards a camera's GUI settings
    from that key, so declaring them under the stale `Settings` silently drops them
    and the processor never sees the forwarded values."""

    def setUp(self):
        with open(
            os.path.join(script_dir, "external_postprocessors.json"), encoding="utf-8"
        ) as handle:
            self.entry = json.load(handle)[0]

    def test_settings_model_declared_under_user_settings_key(self):
        self.assertIn("UserSettings", self.entry)
        self.assertNotIn("Settings", self.entry)


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
        return _make_message(
            {"attributeName": "reviewed", "attributeValue": "true"},
            {"person": [ObjectMetadata()]},
        )

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
        self.assertEqual("true", annotated.objects["person"][0].attributes["reviewed"])

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
