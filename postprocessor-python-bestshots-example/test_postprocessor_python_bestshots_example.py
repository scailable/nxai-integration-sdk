"""Unit tests for the bestshots-example business logic.

External behavior only: a built ObjectsPostProcessorMessage goes in, and the
resulting msg.best_shots is asserted on the way out. No IPC, no sockets. The
`emit_best_shots` config flag lives in a module global the processor sets from
`load_config` at startup; the tests set it directly (same convention as
postprocessor-python-object-counter-threshold's `min_object_count`).
"""

import importlib.util
import os
import sys
import unittest

script_dir = os.path.dirname(os.path.abspath(__file__))
# nxai_postprocessor_sdk lives one level up, in nxai-utilities/
sys.path.insert(0, os.path.join(script_dir, "../nxai-utilities"))

import msgpack
import nxai_postprocessor_sdk
from nxai_postprocessor_sdk import ObjectMetadata, ObjectsPostProcessorMessage

spec = importlib.util.spec_from_file_location(
    "postprocessor_python_bestshots_example",
    os.path.join(script_dir, "postprocessor-python-bestshots-example.py"),
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _tracked_car(track_id="car-track-1", x=0.2, y=0.3, width=0.1, height=0.15):
    obj = ObjectMetadata()
    obj.track_id = track_id
    obj.x = x
    obj.y = y
    obj.width = width
    obj.height = height
    return obj


def _new_car():
    # A car detected for the first time this frame carries new_track_id, not
    # track_id -- not yet an "already-tracked" object.
    obj = ObjectMetadata()
    obj.new_track_id = "car-track-new"
    return obj


# The plugin correlates a response to its request on DeviceID + "|" + Timestamp,
# so both must be non-falsy in the fixture: the SDK defaults are "" and 0, and a
# fixture that left them there would let the echo assertion pass vacuously.
ECHOED_DEVICE_ID = "cam-echo"
ECHOED_TIMESTAMP = 1726000000000000


def _make_message(objects, device_id=ECHOED_DEVICE_ID, timestamp=ECHOED_TIMESTAMP):
    msg = ObjectsPostProcessorMessage()
    msg.device_id = device_id
    msg.timestamp = timestamp
    msg.objects = objects
    return msg


class TestFirstTracked(unittest.TestCase):
    def test_returns_the_first_object_with_a_track_id(self):
        tracked = _tracked_car()
        result = module._first_tracked([_new_car(), tracked])
        self.assertIs(result, tracked)

    def test_returns_none_when_no_object_is_tracked(self):
        self.assertIsNone(module._first_tracked([_new_car()]))

    def test_returns_none_for_empty_list(self):
        self.assertIsNone(module._first_tracked([]))


class TestHandleMessage(unittest.TestCase):
    def setUp(self):
        module.emit_best_shots = False

    def test_disabled_flag_emits_no_best_shot(self):
        module.emit_best_shots = False
        msg = _make_message({"car": [_tracked_car()]})
        module.handle_message(msg)
        self.assertEqual(msg.best_shots, [])

    def test_enabled_flag_tags_the_tracked_car(self):
        module.emit_best_shots = True
        tracked = _tracked_car(track_id="abc-123", x=0.2, y=0.3, width=0.1, height=0.15)
        msg = _make_message({"car": [tracked]})

        module.handle_message(msg)

        self.assertEqual(len(msg.best_shots), 1)
        best_shot = msg.best_shots[0]
        self.assertEqual(best_shot.track_id, "abc-123")

    def test_enabled_flag_uses_the_sentinel_bbox_not_the_objects_own_position(self):
        # Live-verified 2026-07-15: the engine auto-populates every track's
        # best_shot() with a real bbox regardless of any BestShots entry, so
        # copying the object's own per-frame position would be indistinguishable
        # from that auto-population. The sentinel is a synthetic, fixed value a
        # real detector could never plausibly produce.
        module.emit_best_shots = True
        tracked = _tracked_car(track_id="abc-123", x=0.2, y=0.3, width=0.1, height=0.15)
        msg = _make_message({"car": [tracked]})

        module.handle_message(msg)

        best_shot = msg.best_shots[0]
        self.assertEqual(
            (best_shot.x, best_shot.y, best_shot.width, best_shot.height),
            module._SENTINEL_BBOX,
        )

    def test_enabled_flag_with_no_tracked_car_emits_no_best_shot(self):
        module.emit_best_shots = True
        msg = _make_message({"car": [_new_car()]})
        module.handle_message(msg)
        self.assertEqual(msg.best_shots, [])

    def test_enabled_flag_with_no_car_class_emits_no_best_shot(self):
        module.emit_best_shots = True
        msg = _make_message({})
        module.handle_message(msg)
        self.assertEqual(msg.best_shots, [])

    def test_enabled_flag_skips_a_leading_untracked_car_for_a_later_tracked_one(self):
        module.emit_best_shots = True
        tracked = _tracked_car(track_id="second-car")
        msg = _make_message({"car": [_new_car(), tracked]})

        module.handle_message(msg)

        self.assertEqual(len(msg.best_shots), 1)
        self.assertEqual(msg.best_shots[0].track_id, "second-car")


class TestResponseEchoesCorrelationFields(unittest.TestCase):
    def setUp(self):
        module.emit_best_shots = True

    def test_serialized_response_carries_the_requests_device_id_and_timestamp(self):
        # The AI Manager routes a response back to the camera that asked for it by
        # the key DeviceID + "|" + Timestamp. A
        # response that does not echo both fields back unchanged is dropped
        # silently -- no log line anywhere -- so nothing but this assertion
        # catches a processor that answers with a freshly built message.
        # The comparison is against the serialized frame, which is what the
        # plugin actually reads.
        msg = _make_message(
            {"car": [_tracked_car()]}, device_id=ECHOED_DEVICE_ID, timestamp=ECHOED_TIMESTAMP
        )

        result = module.handle_message(msg)

        serialized = msgpack.unpackb(result.to_bytes(), raw=False)
        self.assertEqual(ECHOED_DEVICE_ID, serialized["DeviceID"])
        self.assertEqual(ECHOED_TIMESTAMP, serialized["Timestamp"])


class TestLoadConfig(unittest.TestCase):
    def test_parses_emit_best_shots(self):
        config = nxai_postprocessor_sdk.load_config(
            ["prog", "--emit_best_shots", "true"]
        )
        self.assertEqual(config["emit_best_shots"], "true")


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
        return _make_message({"car": [_tracked_car()]})

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
        module.emit_best_shots = True

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
        self.assertEqual(1, len(annotated.best_shots))

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
