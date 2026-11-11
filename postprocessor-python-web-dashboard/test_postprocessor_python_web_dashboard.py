"""Unit tests for the web-dashboard business logic.

External behavior only: a built ObjectsPostProcessorMessage goes in, and the
derived web-app payload pieces (counts, sizes, positions) are asserted on the
way out, along with the "forwards unmodified" contract. No IPC, no sockets, no
HTTP — the POST side effect is stubbed out.
"""

import importlib.util
import os
import sys
import threading
import unittest
from unittest.mock import patch

import msgpack

script_dir = os.path.dirname(os.path.abspath(__file__))
# nxai_postprocessor_sdk lives one level up, in nxai-utilities/
sys.path.insert(0, os.path.join(script_dir, "../nxai-utilities"))

from nxai_postprocessor_sdk import ObjectMetadata, ObjectsPostProcessorMessage

spec = importlib.util.spec_from_file_location(
    "postprocessor_python_web_dashboard",
    os.path.join(script_dir, "postprocessor-python-web-dashboard.py"),
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _obj(x=0.0, y=0.0, width=0.0, height=0.0, confidence=0.0):
    obj = ObjectMetadata()
    obj.x = x
    obj.y = y
    obj.width = width
    obj.height = height
    obj.confidence = confidence
    return obj


# The plugin correlates a response to its request on DeviceID + "|" + Timestamp,
# so both must be non-falsy in the fixture: the SDK defaults are "" and 0, and a
# fixture that left them there would let the echo assertion pass vacuously.
ECHOED_DEVICE_ID = "cam-echo"
ECHOED_TIMESTAMP = 1726000000000000


def _make_message(objects, device_id="cam-1", timestamp=ECHOED_TIMESTAMP):
    msg = ObjectsPostProcessorMessage()
    msg.objects = objects
    msg.device_id = device_id
    msg.timestamp = timestamp
    return msg


class TestCountObjects(unittest.TestCase):
    def test_counts_via_len_per_class(self):
        msg = _make_message(
            {"person": [_obj(), _obj(), _obj()], "car": [_obj()]}
        )
        self.assertEqual(module.count_objects(msg), {"person": 3, "car": 1})

    def test_no_objects_empty_counts(self):
        self.assertEqual(module.count_objects(_make_message({})), {})


class TestExtractBBoxData(unittest.TestCase):
    def test_size_and_position_derivation(self):
        msg = _make_message({"test": [_obj(x=0.1, y=0.2, width=0.3, height=0.4)]})
        sizes, positions = module.extract_bbox_data(msg)
        # size = (width, height)
        self.assertEqual(sizes, [{"x": 0.3, "y": 0.4, "c": "test"}])
        # pos = (x + width/2, y + height/2)
        self.assertEqual(positions, [{"x": 0.25, "y": 0.4, "c": "test"}])

    def test_rounds_to_four_decimals(self):
        msg = _make_message({"test": [_obj(x=0.123456, y=0.0, width=0.654321, height=0.111119)]})
        sizes, positions = module.extract_bbox_data(msg)
        self.assertEqual(sizes[0]["x"], 0.6543)
        self.assertEqual(sizes[0]["y"], 0.1111)
        # x + width/2 = 0.123456 + 0.3271605 = 0.4506165 -> 0.4506
        self.assertEqual(positions[0]["x"], 0.4506)

    def test_no_objects_empty_lists(self):
        sizes, positions = module.extract_bbox_data(_make_message({}))
        self.assertEqual(sizes, [])
        self.assertEqual(positions, [])


class TestHandleMessage(unittest.TestCase):
    def setUp(self):
        # Stub the fire-and-forget POST so the logic runs without a web app.
        self._calls = []
        self._orig = module.post_to_webapp
        module.post_to_webapp = lambda url, payload: self._calls.append((url, payload))

    def tearDown(self):
        module.post_to_webapp = self._orig

    def test_forwards_message_unmodified(self):
        msg = _make_message({"person": [_obj(x=0.1, confidence=0.9)]})
        out = module.handle_message(msg)
        self.assertIs(out, msg)
        self.assertEqual(list(out.objects), ["person"])
        self.assertEqual(out.events, [])
        self.assertEqual(out.objects["person"][0].attributes, {})

    def test_payload_omits_frame_dimensions(self):
        msg = _make_message({"person": [_obj(x=0.1, width=0.2, height=0.3)]})
        msg.timestamp = 12345
        module.handle_message(msg)
        (_url, payload), = self._calls
        self.assertNotIn("width", payload)
        self.assertNotIn("height", payload)
        self.assertEqual(payload["ts"], 12345)
        self.assertEqual(payload["counts"], {"person": 1})


class TestResponseEchoesCorrelationFields(unittest.TestCase):
    def setUp(self):
        # Stub the fire-and-forget POST so the logic runs without a web app.
        self._orig = module.post_to_webapp
        module.post_to_webapp = lambda url, payload: True

    def tearDown(self):
        module.post_to_webapp = self._orig

    def test_serialized_response_carries_the_requests_device_id_and_timestamp(self):
        # The AI Manager routes a response back to the camera that asked for it by
        # the key DeviceID + "|" + Timestamp. A
        # response that does not echo both fields back unchanged is dropped
        # silently -- no log line anywhere -- so nothing but this assertion
        # catches a processor that answers with a freshly built message.
        # The comparison is against the serialized frame, which is what the
        # plugin actually reads.
        msg = _make_message(
            {"person": [_obj(x=0.1, confidence=0.9)]},
            device_id=ECHOED_DEVICE_ID,
            timestamp=ECHOED_TIMESTAMP,
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
        return _make_message({"person": [_obj(x=0.1, width=0.2, height=0.3)]})

    def write_objects_message(self, msg):
        return b"serialized-response"

    def send_response(self, response_bytes):
        self.responses.append(response_bytes)
        return self._send_succeeds


class TestMessageLoopSurvivability(unittest.TestCase):
    """The engine keeps sending frames; a processor that exits is dropped."""

    def setUp(self):
        # Stub the fire-and-forget POST so the loop runs without a web app.
        self._orig = module.post_to_webapp
        module.post_to_webapp = lambda url, payload: True

    def tearDown(self):
        module.post_to_webapp = self._orig

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
        # The one parseable frame was answered: a loop that skipped it would still
        # reach EXIT with the same call count.
        self.assertEqual([b"serialized-response"], processor.responses)

    def test_unparseable_frame_does_not_end_the_loop(self):
        processor = _FakeProcessor([UNPARSEABLE, b"frame-bytes", EXIT])

        module.run_message_loop(processor)

        self.assertEqual(3, processor.calls, "loop must survive an unparseable frame")
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

    The dashboard's HTTP server and periodic flush live in the companion `web_app.py`,
    a separate program the operator starts; this processor only POSTs to it. What it
    must not do is hold the exit up -- so the wait on that POST is bounded, and the
    tests hold that rather than reading a `daemon=True` off a thread that does not
    exist here. A statically-read flag would not notice a future non-daemon worker.
    """

    def test_the_post_wait_is_bounded(self):
        # An unbounded urlopen is exactly the hazard: a web app that accepts the
        # connection and never answers would hold the loop, and the exit with it,
        # for as long as the OS default allows.
        with patch.object(module.urllib.request, "urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.status = 200

            self.assertTrue(module.post_to_webapp("http://localhost:8111", {"ts": 1}))

        _args, kwargs = urlopen.call_args
        self.assertIn("timeout", kwargs, "urlopen must be given a timeout")
        self.assertLessEqual(kwargs["timeout"], 5)

    def test_an_unreachable_web_app_does_not_end_the_loop(self):
        # The dashboard is a read-only observer: with nothing listening, the frame is
        # still answered to the AI Manager and the loop stays up to see the next EXIT.
        processor = _FakeProcessor([b"frame-bytes", EXIT])

        with patch.object(
            module.urllib.request, "urlopen", side_effect=OSError("connection refused")
        ):
            module.run_message_loop(processor)

        self.assertEqual(2, processor.calls, "an unreachable web app must not end the loop")
        self.assertEqual([b"serialized-response"], processor.responses)

    def test_no_thread_outlives_the_loop(self):
        processor = _FakeProcessor([b"frame-bytes", EXIT])
        before = set(threading.enumerate())

        with patch.object(module.urllib.request, "urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value.status = 200
            module.run_message_loop(processor)

        self.assertTrue(urlopen.called, "the POST path must really have run")
        self.assertEqual(before, set(threading.enumerate()))


if __name__ == "__main__":
    unittest.main()
