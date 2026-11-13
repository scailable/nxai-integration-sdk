"""Unit tests for the height-estimator business logic.

External behavior only: a built ObjectsPostProcessorMessage goes in, and the
per-person Height attribute / fired event are asserted on the way out. No IPC,
no sockets. The calibration knobs are read fresh from `msg.user_settings` on
every call, falling back to DEFAULT_FRAME_FRACTION_PER_METER /
DEFAULT_MAX_HEIGHT_CM when a key is absent.
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
    "postprocessor_python_height_estimator",
    os.path.join(script_dir, "postprocessor-python-height-estimator.py"),
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


# The plugin correlates a response to its request on DeviceID + "|" + Timestamp,
# so both must be non-falsy in the fixture: the SDK defaults are "" and 0, and a
# fixture that left them there would let the echo assertion pass vacuously.
ECHOED_DEVICE_ID = "cam-echo"
ECHOED_TIMESTAMP = 1726000000000000


def _person(height, cls="person"):
    obj = ObjectMetadata()
    obj.height = height
    return obj


def _make_message(objects, user_settings=None, device_id=ECHOED_DEVICE_ID, timestamp=ECHOED_TIMESTAMP):
    msg = ObjectsPostProcessorMessage()
    msg.objects = objects
    msg.user_settings = user_settings or {}
    msg.device_id = device_id
    msg.timestamp = timestamp
    return msg


class TestHeightAttributes(unittest.TestCase):
    def test_height_computed_from_normalized_bbox(self):
        msg = _make_message({"person": [_person(0.5)]})
        module.handle_message(msg)
        # 0.5 / 0.25 == 2.0
        self.assertEqual(msg.objects["person"][0].attributes["Height"], "2.00 m")

    def test_height_formatting_two_decimals(self):
        msg = _make_message({"person": [_person(0.7)]})
        module.handle_message(msg)
        # 0.7 / 0.25 == 2.8
        self.assertEqual(msg.objects["person"][0].attributes["Height"], "2.80 m")

    def test_person_class_matched_case_insensitively(self):
        msg = _make_message({"Person": [_person(0.5)]})
        module.handle_message(msg)
        self.assertEqual(msg.objects["Person"][0].attributes["Height"], "2.00 m")

    def test_attribute_set_on_every_person(self):
        msg = _make_message({"person": [_person(0.4), _person(0.6)]})
        module.handle_message(msg)
        attrs = [o.attributes["Height"] for o in msg.objects["person"]]
        self.assertEqual(attrs, ["1.60 m", "2.40 m"])

    def test_no_persons_no_attributes(self):
        msg = _make_message({"car": [_person(0.5, cls="car")]})
        module.handle_message(msg)
        self.assertEqual(msg.objects["car"][0].attributes, {})

    def test_live_setting_overrides_default_fraction(self):
        msg = _make_message(
            {"person": [_person(0.5)]},
            user_settings={"frame_fraction_per_meter": "0.5"},
        )
        module.handle_message(msg)
        # 0.5 / 0.5 == 1.0, not the default-fraction 2.00 m
        self.assertEqual(msg.objects["person"][0].attributes["Height"], "1.00 m")

    def test_absent_fraction_key_falls_back_to_default(self):
        msg = _make_message({"person": [_person(0.5)]}, user_settings={})
        module.handle_message(msg)
        self.assertEqual(msg.objects["person"][0].attributes["Height"], "2.00 m")


class TestHeightEvent(unittest.TestCase):
    def _fired(self, msg):
        # The emitted id is the bare id from the JSON: the AI Manager copies
        # Events[].ID into the manifest verbatim and relays emitted ids
        # verbatim, so only the bare id survives the mediaserver's check.
        return any(e.type_id == "person.height.exceeded" for e in msg.events)

    def test_below_limit_no_event(self):
        # 0.6 / 0.25 == 2.4 m < 2.5 m
        msg = _make_message({"person": [_person(0.6)]})
        module.handle_message(msg)
        self.assertFalse(self._fired(msg))

    def test_exactly_at_limit_no_event(self):
        # 0.625 / 0.25 == 2.5 m; the event fires strictly above the limit
        msg = _make_message({"person": [_person(0.625)]})
        module.handle_message(msg)
        self.assertFalse(self._fired(msg))

    def test_above_limit_fires(self):
        # 0.7 / 0.25 == 2.8 m > 2.5 m
        msg = _make_message({"person": [_person(0.7)]})
        module.handle_message(msg)
        self.assertTrue(self._fired(msg))

    def test_event_serializes_without_caption(self):
        msg = _make_message({"person": [_person(0.7)]})
        module.handle_message(msg)
        event = next(e for e in msg.events if e.type_id == "person.height.exceeded")
        # v5 drops the v4 Caption: the wire form carries only ID + Description.
        self.assertEqual(set(event.to_dict().keys()), {"ID", "Description"})

    def test_no_persons_no_event(self):
        msg = _make_message({})
        module.handle_message(msg)
        self.assertEqual(msg.events, [])

    def test_live_setting_raises_limit_above_default_height(self):
        # 0.7 / 0.25 == 2.8 m exceeds the hardcoded default limit (2.5 m); a live
        # max_height_cm of 1000 (10 m) suppresses the event instead.
        msg = _make_message(
            {"person": [_person(0.7)]}, user_settings={"max_height_cm": "1000"}
        )
        module.handle_message(msg)
        self.assertFalse(self._fired(msg))

    def test_absent_limit_key_falls_back_to_default(self):
        msg = _make_message({"person": [_person(0.7)]}, user_settings={})
        module.handle_message(msg)
        self.assertTrue(self._fired(msg))


class TestResponseEchoesCorrelationFields(unittest.TestCase):
    def test_serialized_response_carries_the_requests_device_id_and_timestamp(self):
        # The AI Manager routes a response back to the camera that asked for it by
        # the key DeviceID + "|" + Timestamp. A
        # response that does not echo both fields back unchanged is dropped
        # silently -- no log line anywhere -- so nothing but this assertion
        # catches a processor that answers with a freshly built message.
        # The comparison is against the serialized frame, which is what the
        # plugin actually reads.
        msg = _make_message({"person": [_person(0.5)]})

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
        return _make_message({"person": [_person(0.7)]})

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
        # 0.7 / 0.25 == 2.8 m, also above the default limit -> event fired too.
        self.assertEqual(annotated.objects["person"][0].attributes["Height"], "2.80 m")
        self.assertTrue(any(e.type_id == "person.height.exceeded" for e in annotated.events))

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
