"""Unit tests for the CLIP-preprocessor business logic.

External behavior only: a built TensorMessage goes in, and the mutated
"text" tensor is asserted on the way out. No IPC, no sockets. Prompts are
read fresh from `msg.user_settings["prompt1".."prompt5"]` on every frame,
falling back to an empty string per missing prompt.
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
    "preprocessor_python_clip_example",
    os.path.join(script_dir, "preprocessor-python-clip-example.py"),
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _text_tensor():
    tensor = TensorMetadata()
    tensor.name = module.TEXT_TENSOR_NAME
    tensor.shape = [module.PROMPT_COUNT, module.TOKEN_CONTEXT_LENGTH]
    tensor.data_type = nxai_postprocessor_sdk.DATA_TYPE_INT32
    tensor.data = b""
    return tensor


def _message(tensors, user_settings=None):
    msg = TensorMessage()
    msg.tensors = tensors
    msg.user_settings = user_settings or {}
    return msg


def _unpack_tokens(data):
    count = len(data) // 4
    return struct.unpack(f"{count}i", data)


class TestHandleMessage(unittest.TestCase):
    def test_missing_text_tensor_is_appended(self):
        msg = _message({}, user_settings={"prompt1": "a red car"})
        result = module.handle_message(msg)

        self.assertIn(module.TEXT_TENSOR_NAME, result.tensors)
        appended = result.tensors[module.TEXT_TENSOR_NAME]
        self.assertEqual(appended.name, module.TEXT_TENSOR_NAME)
        self.assertEqual(appended.shape, [module.PROMPT_COUNT, module.TOKEN_CONTEXT_LENGTH])
        self.assertEqual(appended.data_type, nxai_postprocessor_sdk.DATA_TYPE_INT32)
        self.assertEqual(
            len(_unpack_tokens(appended.data)),
            module.PROMPT_COUNT * module.TOKEN_CONTEXT_LENGTH,
        )

    def test_appended_text_tensor_carries_the_forwarded_prompts(self):
        msg_empty = _message({})
        module.handle_message(msg_empty)

        msg_with_prompts = _message({}, user_settings={"prompt1": "a red car"})
        module.handle_message(msg_with_prompts)

        self.assertNotEqual(
            msg_empty.tensors[module.TEXT_TENSOR_NAME].data,
            msg_with_prompts.tensors[module.TEXT_TENSOR_NAME].data,
        )

    def test_leaves_other_tensors_untouched(self):
        other = TensorMetadata()
        other.name = "image-"
        other.data = b"\x01\x02"
        msg = _message({"image-": other})
        module.handle_message(msg)
        self.assertEqual(msg.tensors["image-"].data, b"\x01\x02")

    def test_no_settings_forwarded_tokenizes_five_empty_prompts(self):
        msg = _message({module.TEXT_TENSOR_NAME: _text_tensor()})
        module.handle_message(msg)
        tokens = _unpack_tokens(msg.tensors[module.TEXT_TENSOR_NAME].data)
        self.assertEqual(len(tokens), module.PROMPT_COUNT * module.TOKEN_CONTEXT_LENGTH)

    def test_forwarded_prompts_change_the_tokenized_output(self):
        msg_empty = _message({module.TEXT_TENSOR_NAME: _text_tensor()})
        module.handle_message(msg_empty)
        empty_tokens = msg_empty.tensors[module.TEXT_TENSOR_NAME].data

        msg_with_prompts = _message(
            {module.TEXT_TENSOR_NAME: _text_tensor()},
            user_settings={"prompt1": "a red car", "prompt2": "a person walking"},
        )
        module.handle_message(msg_with_prompts)
        prompt_tokens = msg_with_prompts.tensors[module.TEXT_TENSOR_NAME].data

        self.assertNotEqual(empty_tokens, prompt_tokens)

    def test_unexpected_shape_is_a_logged_noop(self):
        tensor = _text_tensor()
        tensor.shape = [module.TOKEN_CONTEXT_LENGTH, module.PROMPT_COUNT]
        msg = _message({module.TEXT_TENSOR_NAME: tensor}, user_settings={"prompt1": "a red car"})

        with self.assertLogs(module.logger, level="ERROR") as logs:
            module.handle_message(msg)

        self.assertEqual(msg.tensors[module.TEXT_TENSOR_NAME].data, b"")
        self.assertIn(str(tensor.shape), logs.output[0])

    def test_unexpected_data_type_is_a_logged_noop(self):
        tensor = _text_tensor()
        tensor.data_type = nxai_postprocessor_sdk.DATA_TYPE_INT64
        msg = _message({module.TEXT_TENSOR_NAME: tensor}, user_settings={"prompt1": "a red car"})

        with self.assertLogs(module.logger, level="ERROR") as logs:
            module.handle_message(msg)

        self.assertEqual(msg.tensors[module.TEXT_TENSOR_NAME].data, b"")
        self.assertIn(str(nxai_postprocessor_sdk.DATA_TYPE_INT64), logs.output[0])

    def test_missing_prompts_default_to_empty_string(self):
        msg = _message(
            {module.TEXT_TENSOR_NAME: _text_tensor()},
            user_settings={"prompt1": "a red car"},
        )
        module.handle_message(msg)
        tokens = _unpack_tokens(msg.tensors[module.TEXT_TENSOR_NAME].data)
        self.assertEqual(len(tokens), module.PROMPT_COUNT * module.TOKEN_CONTEXT_LENGTH)


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

    def parse_tensor_message(self, message_bytes):
        if message_bytes == UNPARSEABLE:
            return None
        return _message({module.TEXT_TENSOR_NAME: _text_tensor()}, user_settings={"prompt1": "a red car"})

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
        tokens = _unpack_tokens(annotated.tensors[module.TEXT_TENSOR_NAME].data)
        self.assertEqual(len(tokens), module.PROMPT_COUNT * module.TOKEN_CONTEXT_LENGTH)

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
