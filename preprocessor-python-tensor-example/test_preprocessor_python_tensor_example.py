"""Unit tests for the tensor-preprocessor business logic.

External behavior only: a built TensorMessage goes in, and the mutated
"nms_sensitivity-" tensor is asserted on the way out. No IPC, no sockets. The
override is read fresh from `msg.user_settings["nmsoverride"]` on every
frame, falling back to `DEFAULT_NMS_OVERRIDE` when the engine
forwards no settings.
"""

import importlib.util
import os
import struct
import sys
import unittest

script_dir = os.path.dirname(os.path.abspath(__file__))
# nxai_postprocessor_sdk lives one level up, in nxai-utilities/
sys.path.insert(0, os.path.join(script_dir, "../nxai-utilities"))

import nxai_postprocessor_sdk
from nxai_postprocessor_sdk import TensorMessage, TensorMetadata

spec = importlib.util.spec_from_file_location(
    "preprocessor_python_tensor_example",
    os.path.join(script_dir, "preprocessor-python-tensor-example.py"),
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _float_tensor(name, value):
    tensor = TensorMetadata()
    tensor.name = name
    tensor.shape = [1]
    tensor.data_type = nxai_postprocessor_sdk.DATA_TYPE_FLOAT
    tensor.data = struct.pack("f", value)
    return tensor


def _message(tensors, user_settings=None):
    msg = TensorMessage()
    msg.tensors = tensors
    msg.user_settings = user_settings or {}
    return msg


class TestHandleMessage(unittest.TestCase):
    def test_overrides_nms_sensitivity_tensor_with_default_when_no_settings_forwarded(self):
        msg = _message({module.NMS_TENSOR_NAME: _float_tensor(module.NMS_TENSOR_NAME, 0.5)})
        module.handle_message(msg)
        (value,) = struct.unpack("f", msg.tensors[module.NMS_TENSOR_NAME].data)
        self.assertAlmostEqual(value, module.DEFAULT_NMS_OVERRIDE, places=5)

    def test_custom_override_value_from_forwarded_settings(self):
        msg = _message(
            {module.NMS_TENSOR_NAME: _float_tensor(module.NMS_TENSOR_NAME, 0.5)},
            user_settings={"nmsoverride": "0.3"},
        )
        module.handle_message(msg)
        (value,) = struct.unpack("f", msg.tensors[module.NMS_TENSOR_NAME].data)
        self.assertAlmostEqual(value, 0.3, places=5)

    def test_leaves_other_tensors_untouched(self):
        msg = _message({"input": _float_tensor("input", 1.0)})
        module.handle_message(msg)
        self.assertEqual(msg.tensors["input"].data, struct.pack("f", 1.0))

    def test_missing_nms_tensor_is_noop(self):
        msg = _message({})
        result = module.handle_message(msg)
        self.assertEqual(result.tensors, {})


EXIT = object()  # Sentinel script item: simulates IpcSignal.EXIT via exit_requested.
UNPARSEABLE = b"unparseable-frame"  # Script item for which parse_tensor_message returns None.


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
    frame that was received but never processed and returned is visible too.
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

    def parse_tensor_message(self, message_bytes):
        if message_bytes == UNPARSEABLE:
            return None
        return _message(
            {module.NMS_TENSOR_NAME: _float_tensor(module.NMS_TENSOR_NAME, 0.5)},
            user_settings={"nmsoverride": "0.3"},
        )

    def write_tensor_message(self, msg):
        # Stand in for msgpack serialization: hand back the mutated message so the
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
        (value,) = struct.unpack("f", annotated.tensors[module.NMS_TENSOR_NAME].data)
        self.assertAlmostEqual(value, 0.3, places=5)

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
