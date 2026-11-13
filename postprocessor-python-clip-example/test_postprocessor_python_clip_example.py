"""Unit tests for the CLIP-postprocessor business logic.

External behavior only: a parsed `ObjectsPostProcessorMessage` (with its
`tensors` map) goes in, and the mutated message (cached attributes, newly cached
prompt, emitted events) is asserted on the way out. No IPC, no sockets.
"""

import importlib.util
import os
import struct
import sys
import unittest

import msgpack

script_dir = os.path.dirname(os.path.abspath(__file__))
_nxai_utilities = os.path.join(script_dir, "../nxai-utilities")
if _nxai_utilities not in sys.path:
    sys.path.insert(0, _nxai_utilities)

import nxai_postprocessor_sdk
from nxai_postprocessor_sdk import ObjectMetadata, ObjectsPostProcessorMessage, TensorMetadata

spec = importlib.util.spec_from_file_location(
    "postprocessor_python_clip_example",
    os.path.join(script_dir, "postprocessor-python-clip-example.py"),
)
module = importlib.util.module_from_spec(spec)
sys.modules["postprocessor_python_clip_example"] = module
spec.loader.exec_module(module)


def _scores_bytes(values):
    return struct.pack(f"<{len(values)}f", *values)


def _scores_tensor(values, shape=None, data_type=None):
    return {
        "Name": module.SCORES_TENSOR_NAME,
        "DataType": nxai_postprocessor_sdk.DATA_TYPE_FLOAT if data_type is None else data_type,
        "Shape": module.EXPECTED_SCORES_SHAPE if shape is None else shape,
        "Data": _scores_bytes(values),
    }


# The plugin correlates a response to its request on DeviceID + "|" + Timestamp,
# so both must be non-falsy in the fixture: the SDK defaults are "" and 0, and a
# fixture that left them there would let the echo assertion pass vacuously.
ECHOED_DEVICE_ID = "cam-echo"
ECHOED_TIMESTAMP = 1726000000000000


def _message(
    objects=None,
    user_settings=None,
    original_object_id=None,
    tensors=None,
    device_id="cam-1",
    timestamp=ECHOED_TIMESTAMP,
):
    msg = ObjectsPostProcessorMessage()
    msg.device_id = device_id
    msg.timestamp = timestamp
    msg.objects = objects or {}
    msg.user_settings = user_settings or {}
    msg.original_object_id = original_object_id
    for tensor in tensors or []:
        msg.tensors[tensor["Name"]] = TensorMetadata.from_msgpack_pair(
            tensor["Name"], tensor["Data"], tensor["Shape"], tensor["DataType"]
        )
    return msg


def _prompts(**kwargs):
    settings = {f"prompt{i}": "" for i in range(1, module.PROMPT_COUNT + 1)}
    settings.update(kwargs)
    return settings


class TestScoresTensorGuard(unittest.TestCase):
    def setUp(self):
        module.objects_attributes.clear()
        module.previous_state.clear()
        module.previous_event_time.clear()

    def test_absent_tensors_field_is_a_debug_noop(self):
        msg = _message(user_settings=_prompts(prompt1="a red car"))
        with self.assertLogs(module.logger, level="DEBUG") as logs:
            result = module.handle_message(msg)
        self.assertEqual(result.events, [])
        self.assertIn("DEBUG", logs.output[0])

    def test_absent_scores_entry_is_a_debug_noop(self):
        msg = _message(
            user_settings=_prompts(prompt1="a red car"),
            tensors=[{"Name": "some_other_tensor", "DataType": 1, "Shape": [1], "Data": b""}],
        )
        with self.assertLogs(module.logger, level="DEBUG") as logs:
            result = module.handle_message(msg)
        self.assertEqual(result.events, [])
        self.assertIn("DEBUG", logs.output[0])

    def test_wrong_shape_is_an_error_noop(self):
        msg = _message(
            user_settings=_prompts(prompt1="a red car"),
            tensors=[_scores_tensor([0.1, 0.2, 0.3, 0.4, 0.5], shape=[1, 3])],
        )
        with self.assertLogs(module.logger, level="ERROR") as logs:
            result = module.handle_message(msg)
        self.assertEqual(result.events, [])
        self.assertIn(str(module.EXPECTED_SCORES_SHAPE), logs.output[0])
        self.assertIn("[1, 3]", logs.output[0])

    def test_wrong_dtype_is_an_error_noop(self):
        msg = _message(
            user_settings=_prompts(prompt1="a red car"),
            tensors=[_scores_tensor([0.1, 0.2, 0.3, 0.4, 0.5], data_type=nxai_postprocessor_sdk.DATA_TYPE_INT32)],
        )
        with self.assertLogs(module.logger, level="ERROR") as logs:
            result = module.handle_message(msg)
        self.assertEqual(result.events, [])
        self.assertIn(str(nxai_postprocessor_sdk.DATA_TYPE_INT32), logs.output[0])

    def test_correct_tensor_scores_normally(self):
        msg = _message(
            user_settings=_prompts(prompt1="a red car"),
            tensors=[_scores_tensor([0.9, 0.1, 0.1, 0.1, 0.1])],
            original_object_id="obj-1",
        )
        module.handle_message(msg)
        self.assertEqual(module.objects_attributes["obj-1"], "a red car")

    def test_scores_are_read_through_msg_tensors_of_a_parsed_frame(self):
        frame = msgpack.packb(
            {
                "DeviceID": "cam-1",
                "Timestamp": ECHOED_TIMESTAMP,
                "OriginalObjectID": "obj-wire",
                "UserSettings": _prompts(prompt1="a red car"),
                "Tensors": [_scores_tensor([0.9, 0.1, 0.1, 0.1, 0.1])],
            }
        )
        msg = ObjectsPostProcessorMessage.from_dict(msgpack.unpackb(frame, raw=False))
        self.assertIn(module.SCORES_TENSOR_NAME, msg.tensors)
        module.handle_message(msg)
        self.assertEqual(module.objects_attributes["obj-wire"], "a red car")


class TestEmptySlotExclusion(unittest.TestCase):
    def setUp(self):
        module.objects_attributes.clear()
        module.previous_state.clear()
        module.previous_event_time.clear()

    def test_empty_slot_never_wins_even_with_the_highest_score(self):
        msg = _message(
            user_settings=_prompts(prompt1="a red car"),  # prompt2..5 stay empty
            tensors=[_scores_tensor([0.1, 0.9, 0.1, 0.1, 0.1])],  # slot 2 (empty) scores highest
            original_object_id="obj-1",
        )
        module.handle_message(msg)
        self.assertEqual(module.objects_attributes["obj-1"], "a red car")

    def test_all_slots_empty_is_a_noop(self):
        msg = _message(
            user_settings=_prompts(),
            tensors=[_scores_tensor([0.9, 0.8, 0.7, 0.6, 0.5])],
            original_object_id="obj-1",
        )
        module.handle_message(msg)
        self.assertNotIn("obj-1", module.objects_attributes)


class TestScoreCache(unittest.TestCase):
    def setUp(self):
        module.objects_attributes.clear()
        module.previous_state.clear()
        module.previous_event_time.clear()

    def test_cache_evicts_fifo_past_100_entries(self):
        for i in range(100):
            module._cache_winning_prompt(f"obj-{i}", "a red car")
        self.assertEqual(len(module.objects_attributes), 100)

        module._cache_winning_prompt("obj-100", "a red car")

        self.assertEqual(len(module.objects_attributes), 100)
        self.assertNotIn("obj-0", module.objects_attributes)
        self.assertIn("obj-1", module.objects_attributes)
        self.assertIn("obj-100", module.objects_attributes)

    def test_cached_prompt_applied_to_detector_object_as_stable_attribute(self):
        module.objects_attributes["obj-1"] = "a red car"
        obj = ObjectMetadata()
        obj.track_id = "obj-1"
        msg = _message(objects={"person": [obj]})

        module.handle_message(msg)

        self.assertEqual(obj.attributes[module.CLIP_PROMPT_ATTRIBUTE], "a red car")

    def test_uncached_object_is_untouched(self):
        obj = ObjectMetadata()
        obj.track_id = "obj-unknown"
        msg = _message(objects={"person": [obj]})

        module.handle_message(msg)

        self.assertNotIn(module.CLIP_PROMPT_ATTRIBUTE, obj.attributes)


class TestScoringAndEvents(unittest.TestCase):
    def setUp(self):
        module.objects_attributes.clear()
        module.previous_state.clear()
        module.previous_event_time.clear()

    def test_argmax_picks_the_highest_scoring_non_empty_prompt(self):
        result = module._argmax_prompt(
            [0.1, 0.9, 0.5, 0.2, 0.3],
            {1: "a red car", 2: "a person walking", 3: "a bicycle"},
        )
        self.assertEqual(result, ("a person walking", 0.9))

    def test_emits_event_with_prefixed_id_and_winning_prompt(self):
        msg = _message(
            user_settings=_prompts(prompt1="a red car"),
            tensors=[_scores_tensor([0.9, 0.1, 0.1, 0.1, 0.1])],
        )
        module.handle_message(msg)
        self.assertEqual(len(msg.events), 1)
        self.assertEqual(msg.events[0].type_id, "clip.event")
        self.assertEqual(msg.events[0].description, "a red car")

    def test_event_cooldown_suppresses_repeated_identical_events(self):
        msg = _message(
            user_settings=_prompts(prompt1="a red car", event_cooldown_sec="60"),
            tensors=[_scores_tensor([0.9, 0.1, 0.1, 0.1, 0.1])],
        )
        module.handle_message(msg)

        msg2 = _message(
            user_settings=_prompts(prompt1="a red car", event_cooldown_sec="60"),
            tensors=[_scores_tensor([0.9, 0.1, 0.1, 0.1, 0.1])],
        )
        module.handle_message(msg2)

        self.assertEqual(len(msg2.events), 0)

    def test_different_winning_prompt_emits_even_within_cooldown(self):
        msg = _message(
            user_settings=_prompts(prompt1="a red car", event_cooldown_sec="60"),
            tensors=[_scores_tensor([0.9, 0.1, 0.1, 0.1, 0.1])],
        )
        module.handle_message(msg)

        msg2 = _message(
            user_settings=_prompts(prompt2="a person walking", event_cooldown_sec="60"),
            tensors=[_scores_tensor([0.1, 0.9, 0.1, 0.1, 0.1])],
        )
        module.handle_message(msg2)

        self.assertEqual(len(msg2.events), 1)
        self.assertEqual(msg2.events[0].description, "a person walking")


class TestResponseEchoesCorrelationFields(unittest.TestCase):
    def setUp(self):
        module.objects_attributes.clear()
        module.previous_state.clear()
        module.previous_event_time.clear()

    def test_serialized_response_carries_the_requests_device_id_and_timestamp(self):
        # The AI Manager routes a response back to the camera that asked for it by
        # the key DeviceID + "|" + Timestamp. A
        # response that does not echo both fields back unchanged is dropped
        # silently -- no log line anywhere -- so nothing but this assertion
        # catches a processor that answers with a freshly built message.
        # The comparison is against the serialized frame, which is what the
        # plugin actually reads.
        msg = _message(device_id=ECHOED_DEVICE_ID, timestamp=ECHOED_TIMESTAMP)

        result = module.handle_message(msg)

        serialized = msgpack.unpackb(result.to_bytes(), raw=False)
        self.assertEqual(ECHOED_DEVICE_ID, serialized["DeviceID"])
        self.assertEqual(ECHOED_TIMESTAMP, serialized["Timestamp"])


EXIT = object()  # Sentinel script item: simulates IpcSignal.EXIT via exit_requested.
# An "unparseable" frame is one that unpacks fine as msgpack but fails from_dict():
# "Objects" as a list, not a dict, blows up the `.items()` call in from_dict()'s try
# block, which catches it and returns None -- exactly the `msg is None` branch the
# loop checks for.
FRAME = msgpack.packb({}, use_bin_type=True)
UNPARSEABLE = msgpack.packb({"Objects": [1, 2, 3]}, use_bin_type=True)


class _ScriptExhausted(BaseException):
    """Raised when the loop asks for more messages than the script holds.

    Derives from BaseException rather than Exception on purpose. run_message_loop's
    `except Exception` branch ends in `continue` here (not `break`), which would
    otherwise swallow this guard and let the loop spin past the end of the script --
    turning every assertion below into one that cannot fail. An EXIT-swallowing
    regression, exactly the defect these tests exist to catch, would then still pass.
    """


class _FakeProcessor:
    """Replays a scripted sequence of wait_for_message() outcomes.

    `calls` counts wait_for_message() invocations, which is what pins down where the
    loop stopped: asserting on exit_requested alone cannot distinguish "returned on
    EXIT" from "kept spinning past it". `sent` collects the serialized responses, so a
    frame that was received but never annotated and returned is visible too.
    """

    def __init__(self, script, send_response_result=True):
        self._script = list(script)
        self.calls = 0
        self.exit_requested = False
        self.sent = []
        self._send_response_result = send_response_result

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

    def parse_objects_message(self, data):
        return ObjectsPostProcessorMessage.from_bytes(data)

    def write_objects_message(self, msg):
        # Stand in for msgpack serialization: hand back the annotated message so
        # the test can assert the frame really went through handle_message.
        return msg

    def send_response(self, response_bytes):
        self.sent.append(response_bytes)
        return self._send_response_result


class TestMessageLoopSurvivability(unittest.TestCase):
    """The engine keeps sending frames; a processor that exits is dropped."""

    def setUp(self):
        module.objects_attributes.clear()
        module.previous_state.clear()
        module.previous_event_time.clear()

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
        processor = _FakeProcessor([None, None, FRAME, EXIT])

        module.run_message_loop(processor)

        self.assertEqual(4, processor.calls, "loop must survive both timeouts and the frame")
        self.assertTrue(processor.exit_requested)
        # The one frame was actually handled: a loop that skipped it would still
        # reach EXIT with the same call count.
        self.assertEqual(1, len(processor.sent))
        self.assertIsInstance(processor.sent[0], ObjectsPostProcessorMessage)

    def test_unparseable_frame_does_not_end_the_loop(self):
        # A frame that unpacks as msgpack but fails ObjectsPostProcessorMessage.from_dict()
        # is logged and skipped, not fatal.
        processor = _FakeProcessor([UNPARSEABLE, FRAME, EXIT])

        module.run_message_loop(processor)

        self.assertEqual(3, processor.calls, "loop must survive an unparseable frame")
        self.assertEqual(1, len(processor.sent), "only the parseable frame is answered")
        self.assertTrue(processor.exit_requested)

    def test_raising_message_does_not_end_the_loop(self):
        # Unlike the confidences-example loop, this loop's except clause ends in
        # `continue`, not `break`: an unexpected exception from wait_for_message
        # is logged and survived, and the loop goes on to process EXIT normally.
        processor = _FakeProcessor([RuntimeError("transport hiccup"), EXIT])

        with self.assertLogs(module.logger, level="ERROR"):
            module.run_message_loop(processor)

        self.assertEqual(2, processor.calls, "loop must survive the exception and reach EXIT")
        self.assertTrue(processor.exit_requested)

    def test_failed_send_response_ends_the_loop(self):
        # A failed send_response() means the return channel is broken: the loop
        # must give up rather than keep asking for more frames it cannot answer.
        processor = _FakeProcessor([FRAME, EXIT], send_response_result=False)

        with self.assertLogs(module.logger, level="ERROR") as captured:
            module.run_message_loop(processor)

        self.assertEqual(1, processor.calls, "loop must return after the failed send, before EXIT")
        self.assertFalse(processor.exit_requested)
        self.assertTrue(any("Failed to send response" in r.getMessage() for r in captured.records))


if __name__ == "__main__":
    unittest.main()
