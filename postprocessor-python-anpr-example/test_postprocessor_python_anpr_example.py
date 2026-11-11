import unittest
import time
import sys
import os
import json
import re
import importlib.util

import msgpack

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.join(script_dir, "..")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
_nxai_utilities = os.path.join(repo_root, "nxai-utilities")
if _nxai_utilities not in sys.path:
    sys.path.insert(0, _nxai_utilities)

from nxai_postprocessor_sdk import ObjectMetadata, ObjectsPostProcessorMessage
from message_processing_utils.anpr import (
    AnprDetectorMessage,
    CctOcrMessage,
    EventDeduplicationCache,
)
from message_processing_utils.general.ocr import LogitsOcrEngine, OcrCache

spec = importlib.util.spec_from_file_location(
    "postprocessor_python_anpr_example",
    os.path.join(script_dir, "postprocessor-python-anpr-example.py"),
)
module = importlib.util.module_from_spec(spec)
sys.modules["postprocessor_python_anpr_example"] = module
spec.loader.exec_module(module)


def _object(track_id):
    obj = ObjectMetadata()
    obj.track_id = track_id
    return obj


# The plugin correlates a response to its request on DeviceID + "|" + Timestamp,
# so both must be non-falsy in the fixture: the SDK defaults are "" and 0, and a
# fixture that left them there would let the echo assertion pass vacuously.
ECHOED_DEVICE_ID = "cam-echo"
ECHOED_TIMESTAMP = 1726000000000000


def _detector_message(track_id, user_settings=None, device_id=ECHOED_DEVICE_ID, timestamp=ECHOED_TIMESTAMP):
    sdk_msg = ObjectsPostProcessorMessage()
    sdk_msg.objects = {"lp": [_object(track_id)]}
    sdk_msg.user_settings = user_settings or {}
    sdk_msg.device_id = device_id
    sdk_msg.timestamp = timestamp
    return AnprDetectorMessage(sdk_msg)


def _detector_message_bytes(track_id):
    return _detector_message(track_id).to_bytes()


def _logits_for(text, confidence):
    """One-hot (9, 37) logits that decode to `text` at `confidence` per character.

    Positions past the end of `text` decode to the blank character, which the
    engine strips, so an empty `text` decodes to no plate at all.
    """
    import numpy as np

    rows, cols = module.EXPECTED_LOGITS_SHAPE
    logits = np.zeros((rows, cols), dtype=np.float32)
    padded = text.ljust(rows)
    for position, char in enumerate(padded):
        logits[position, module.CHAR_MAP.index(char)] = confidence
    return logits


def _ocr_message(
    original_object_id,
    text=None,
    confidence=0.99,
    device_id=ECHOED_DEVICE_ID,
    timestamp=ECHOED_TIMESTAMP,
):
    """A message from the chained OCR model, as the engine sends it.

    With `text` None the message carries no Tensors block, the same as the
    engine sends when ReceiveBinaryData is off.
    """
    sdk_msg = ObjectsPostProcessorMessage()
    sdk_msg.original_object_id = original_object_id
    sdk_msg.device_id = device_id
    sdk_msg.timestamp = timestamp
    if text is None:
        return AnprDetectorMessage(sdk_msg)
    raw_tensors = [
        {"Name": module.DEFAULT_OCR_OUTPUT_NAME, "Data": _logits_for(text, confidence).tobytes()}
    ]
    return CctOcrMessage(sdk_msg, raw_tensors)


def _ocr_cache():
    engine = LogitsOcrEngine(
        expected_logits_shape=module.EXPECTED_LOGITS_SHAPE, char_map=module.CHAR_MAP
    )
    return OcrCache(engine, module.DEFAULT_OCR_OUTPUT_NAME)


def _handle(message, ocr_cache, event_cache=None, min_confidence=0.95, **kwargs):
    """Run handle_message and return the serialized response, which is what the plugin reads."""
    result = module.handle_message(
        message, ocr_cache, event_cache or EventDeduplicationCache(), min_confidence, **kwargs
    )
    return msgpack.unpackb(result.to_bytes(), raw=False)


def _plate_entry(serialized, class_name=module.DEFAULT_PLATE_CLASS_NAME):
    objects = serialized["Objects"][class_name]
    assert len(objects) == 1, objects
    return objects[0]


class TestHandleMessageOcr(unittest.TestCase):
    def test_sets_plate_attributes_on_the_original_object(self):
        response = _handle(_ocr_message("obj-1", "ABC123", confidence=0.97), _ocr_cache())

        entry = _plate_entry(response)
        self.assertEqual(entry["ID"], "obj-1")
        self.assertEqual(entry["attributes"]["License Plate Text"], "ABC123")
        self.assertEqual(entry["attributes"]["Confidence"], "0.9700")
        self.assertNotIn("removeAttributes", entry)

    def test_update_is_attribute_only(self):
        # The processor only knows the original object's ID: sending a bbox or
        # confidence would overwrite the detector's real values with zeroes.
        response = _handle(_ocr_message("obj-1", "ABC123"), _ocr_cache())

        entry = _plate_entry(response)
        for key in ("x", "y", "width", "height", "confidence", "NewID"):
            self.assertNotIn(key, entry)

    def test_blank_reading_removes_plate_attributes(self):
        response = _handle(_ocr_message("obj-1", ""), _ocr_cache())

        entry = _plate_entry(response)
        self.assertEqual(entry["ID"], "obj-1")
        self.assertEqual(entry["attributes"], {})
        self.assertEqual(
            sorted(entry["removeAttributes"]), sorted(["License Plate Text", "Confidence"])
        )

    def test_missing_tensors_removes_plate_attributes(self):
        response = _handle(_ocr_message("obj-1", text=None), _ocr_cache())

        entry = _plate_entry(response)
        self.assertEqual(
            sorted(entry["removeAttributes"]), sorted(["License Plate Text", "Confidence"])
        )

    def test_keeps_the_best_reading_for_the_plate(self):
        ocr_cache = _ocr_cache()
        _handle(_ocr_message("obj-1", "ABC123", confidence=0.98), ocr_cache)

        response = _handle(_ocr_message("obj-1", "ABC128", confidence=0.60), ocr_cache)

        attributes = _plate_entry(response)["attributes"]
        self.assertEqual(attributes["License Plate Text"], "ABC123")
        self.assertEqual(attributes["Confidence"], "0.9800")

    def test_blank_crop_after_a_reading_keeps_the_reading(self):
        # One unreadable crop (motion blur, occlusion) must not wipe a plate
        # that was already read for this object.
        ocr_cache = _ocr_cache()
        _handle(_ocr_message("obj-1", "ABC123"), ocr_cache)

        response = _handle(_ocr_message("obj-1", ""), ocr_cache)

        entry = _plate_entry(response)
        self.assertEqual(entry["attributes"]["License Plate Text"], "ABC123")
        self.assertNotIn("removeAttributes", entry)

    def test_readings_are_per_object(self):
        ocr_cache = _ocr_cache()
        _handle(_ocr_message("obj-1", "ABC123"), ocr_cache)

        response = _handle(_ocr_message("obj-2", ""), ocr_cache)

        entry = _plate_entry(response)
        self.assertEqual(entry["ID"], "obj-2")
        self.assertIn("License Plate Text", entry["removeAttributes"])

    def test_returns_the_plate_under_the_configured_class(self):
        response = _handle(
            _ocr_message("obj-1", "ABC123"), _ocr_cache(), plate_class_name="plate"
        )

        self.assertEqual(list(response["Objects"]), ["plate"])
        self.assertEqual(_plate_entry(response, "plate")["ID"], "obj-1")

    def test_message_without_original_object_is_returned_unchanged(self):
        # The processor is not meant to be assigned to the detector; if it is,
        # its objects go back exactly as received.
        message = _detector_message("obj-1")
        before = msgpack.unpackb(message.to_bytes(), raw=False)

        response = _handle(message, _ocr_cache())

        self.assertEqual(response, before)


class TestPlateEvents(unittest.TestCase):
    def test_emits_event_above_threshold(self):
        response = _handle(_ocr_message("obj-1", "ABC123", confidence=0.97), _ocr_cache())

        self.assertEqual(
            response["Events"],
            [{"ID": module.EVENT_ID_PLATE_RECOGNIZED, "Description": "ABC123"}],
        )

    def test_no_event_below_threshold(self):
        response = _handle(_ocr_message("obj-1", "ABC123", confidence=0.80), _ocr_cache())

        self.assertEqual(response["Events"], [])
        # The reading is still attached; the threshold only gates the event.
        self.assertEqual(_plate_entry(response)["attributes"]["License Plate Text"], "ABC123")

    def test_no_event_when_no_plate(self):
        response = _handle(_ocr_message("obj-1", ""), _ocr_cache())

        self.assertEqual(response["Events"], [])

    def test_no_duplicate_event_same_object_same_text(self):
        ocr_cache = _ocr_cache()
        event_cache = EventDeduplicationCache()

        first = _handle(_ocr_message("obj-1", "XYZ999"), ocr_cache, event_cache)
        second = _handle(_ocr_message("obj-1", "XYZ999"), ocr_cache, event_cache)

        self.assertEqual(len(first["Events"]), 1)
        self.assertEqual(second["Events"], [])

    def test_new_event_when_plate_text_changes(self):
        ocr_cache = _ocr_cache()
        event_cache = EventDeduplicationCache()

        first = _handle(_ocr_message("obj-1", "ABC123", confidence=0.96), ocr_cache, event_cache)
        second = _handle(_ocr_message("obj-1", "DEF456", confidence=0.99), ocr_cache, event_cache)

        self.assertEqual(len(first["Events"]), 1)
        self.assertEqual(len(second["Events"]), 1)
        self.assertEqual(second["Events"][0]["Description"], "DEF456")


class TestResponseEchoesCorrelationFields(unittest.TestCase):
    def test_serialized_response_carries_the_requests_device_id_and_timestamp(self):
        # The AI Manager routes a response back to the camera that asked for it by
        # the key DeviceID + "|" + Timestamp. A
        # response that does not echo both fields back unchanged is dropped
        # silently -- no log line anywhere -- so nothing but this assertion
        # catches a processor that answers with a freshly built message.
        # The comparison is against the serialized frame, which is what the
        # plugin actually reads.
        message = _ocr_message(
            "obj-1", "ABC123", device_id=ECHOED_DEVICE_ID, timestamp=ECHOED_TIMESTAMP
        )

        serialized = _handle(message, _ocr_cache())

        self.assertEqual(ECHOED_DEVICE_ID, serialized["DeviceID"])
        self.assertEqual(ECHOED_TIMESTAMP, serialized["Timestamp"])


class TestMessageParsing(unittest.TestCase):
    def test_engine_ocr_message_parses_and_gets_plate(self):
        # The full path from the bytes the engine sends: OriginalObjectID plus a
        # Tensors array of {Name, DataType, Shape, Data} maps.
        logits = _logits_for("ABC123", 0.99)
        engine_message = msgpack.packb(
            {
                "Timestamp": ECHOED_TIMESTAMP,
                "DeviceID": ECHOED_DEVICE_ID,
                "DeviceName": "cam",
                "Objects": {},
                "Events": [],
                "OriginalObjectID": "obj-1",
                "Tensors": [
                    {
                        "Name": module.DEFAULT_OCR_OUTPUT_NAME,
                        "DataType": 1,
                        "Shape": [1, *module.EXPECTED_LOGITS_SHAPE],
                        "Data": logits.tobytes(),
                    }
                ],
            },
            use_bin_type=True,
        )

        message = module.create_anpr_message_from_bytes(engine_message)
        self.assertIsInstance(message, CctOcrMessage)

        response = _handle(message, _ocr_cache())
        self.assertEqual(
            _plate_entry(response)["attributes"]["License Plate Text"], "ABC123"
        )


class TestMinConfidenceUserSettings(unittest.TestCase):
    def test_reads_forwarded_min_confidence(self):
        message = _detector_message("obj-1", user_settings={"min_confidence": "0.5"})
        self.assertEqual(
            float(message.user_settings.get("min_confidence", module.DEFAULT_MIN_CONFIDENCE)),
            0.5,
        )

    def test_falls_back_to_default_when_absent(self):
        message = _detector_message("obj-1")
        self.assertEqual(
            float(message.user_settings.get("min_confidence", module.DEFAULT_MIN_CONFIDENCE)),
            module.DEFAULT_MIN_CONFIDENCE,
        )


class TestRegistrationJson(unittest.TestCase):
    def test_declares_receive_binary_data(self):
        json_path = os.path.join(script_dir, "external_postprocessors.json")
        with open(json_path, encoding="utf-8") as handle:
            entries = json.load(handle)

        self.assertTrue(
            entries[0].get("ReceiveBinaryData") is True,
            "external_postprocessors.json must set ReceiveBinaryData: true — "
            "the OCR path reads Tensors (base/messages.py get_binary_output_f32)",
        )

    def test_readme_snippet_matches_registration_json(self):
        """The README snippet is what operators copy — it must not drift."""
        with open(os.path.join(script_dir, "external_postprocessors.json"), encoding="utf-8") as handle:
            entry = json.load(handle)[0]
        with open(os.path.join(script_dir, "README.md"), encoding="utf-8") as handle:
            readme = handle.read()

        blocks = re.findall(r"```json\n(.*?)```", readme, re.DOTALL)
        self.assertEqual(len(blocks), 1, "README must carry exactly one json snippet")

        self.assertEqual(json.loads(blocks[0]), entry)

    def test_emits_the_event_id_exactly_as_declared(self):
        """The mediaserver drops an emitted id that differs from Events[].ID."""
        with open(os.path.join(script_dir, "external_postprocessors.json"), encoding="utf-8") as handle:
            entry = json.load(handle)[0]

        declared = [event["ID"] for event in entry["Events"]]
        self.assertEqual(declared, ["anpr.plate_recognized"])
        self.assertEqual(module.EVENT_ID_PLATE_RECOGNIZED, declared[0])


EXIT = object()  # Sentinel script item: simulates IpcSignal.EXIT via exit_requested.


class _FakeProcessor:
    """Replays a scripted sequence of wait_for_message() outcomes.

    A `StopIteration`-terminated script would be swallowed by the loop's own
    except clause, so the script ends with KeyboardInterrupt, which the loop
    deliberately does not catch.
    """

    def __init__(self, script, send_results=None):
        self._script = list(script)
        self._send_results = list(send_results or [])
        self.sent = []
        self.exit_requested = False

    def wait_for_message(self, timeout_ms=None):
        if not self._script:
            raise KeyboardInterrupt
        item = self._script.pop(0)
        if item is EXIT:
            self.exit_requested = True
            return None
        if isinstance(item, Exception):
            raise item
        return item

    def send_response(self, data):
        self.sent.append(data)
        if self._send_results:
            return self._send_results.pop(0)
        return True


class TestMessageLoopSurvivability(unittest.TestCase):
    """The engine keeps sending frames; a processor that exits is dropped."""

    def _caches(self):
        return (
            OcrCache(engine=None, output_name=module.DEFAULT_OCR_OUTPUT_NAME),
            EventDeduplicationCache(ttl_sec=300.0),
        )

    def _run(self, processor):
        ocr_cache, event_cache = self._caches()
        with self.assertRaises(KeyboardInterrupt):
            module.run_message_loop(processor, ocr_cache, event_cache)

    def test_unparsable_message_does_not_end_the_loop(self):
        good = _detector_message_bytes("obj-1")
        processor = _FakeProcessor([b"\xc1not-msgpack", good])
        self._run(processor)
        self.assertEqual(len(processor.sent), 1, "the message after a bad one must still be served")

    def test_raising_message_does_not_end_the_loop(self):
        good = _detector_message_bytes("obj-1")
        processor = _FakeProcessor([RuntimeError("transport hiccup"), good])
        self._run(processor)
        self.assertEqual(len(processor.sent), 1)

    def test_timeout_does_not_end_the_loop(self):
        good = _detector_message_bytes("obj-1")
        processor = _FakeProcessor([None, None, good])
        self._run(processor)
        self.assertEqual(len(processor.sent), 1)

    def test_exit_signal_ends_the_loop_cleanly(self):
        # A graceful EXIT must return, not be swallowed as a plain timeout
        # that keeps the loop spinning until the AI Manager kills the process.
        processor = _FakeProcessor([EXIT, _detector_message_bytes("obj-1")])
        ocr_cache, event_cache = self._caches()
        module.run_message_loop(processor, ocr_cache, event_cache)
        self.assertEqual(len(processor.sent), 0, "loop must stop before serving the next message")

    def test_failed_send_ends_the_loop(self):
        good = _detector_message_bytes("obj-1")
        processor = _FakeProcessor([good, good], send_results=[False])
        ocr_cache, event_cache = self._caches()
        # Returns instead of raising: the script's second message is never read.
        module.run_message_loop(processor, ocr_cache, event_cache)
        self.assertEqual(len(processor.sent), 1)

    def test_handler_failure_still_sends_unmodified_response(self):
        # handle_message() raises after the message was parsed successfully;
        # the engine still expects a response for that frame.
        bad_confidence = _detector_message_bytes("obj-1")
        good = _detector_message_bytes("obj-2")
        processor = _FakeProcessor([bad_confidence, good])

        ocr_cache, event_cache = self._caches()
        original_handle_message = module.handle_message

        def _boom(message, *args, **kwargs):
            raise RuntimeError("decode blew up")

        module.handle_message = _boom
        try:
            with self.assertRaises(KeyboardInterrupt):
                module.run_message_loop(processor, ocr_cache, event_cache)
        finally:
            module.handle_message = original_handle_message

        self.assertEqual(len(processor.sent), 2, "the failed frame must still get a response")


class TestMinConfidenceFallback(unittest.TestCase):
    """A non-numeric forwarded UserSettings value must not kill every frame."""

    def test_garbage_min_confidence_falls_back_to_default(self):
        result = module._get_float_setting({"min_confidence": "not-a-number"}, "min_confidence", module.DEFAULT_MIN_CONFIDENCE)
        self.assertEqual(result, module.DEFAULT_MIN_CONFIDENCE)

    def test_garbage_min_confidence_does_not_kill_the_loop(self):
        sdk_msg = ObjectsPostProcessorMessage()
        sdk_msg.objects = {"lp": [_object("obj-1")]}
        sdk_msg.user_settings = {"min_confidence": "garbage"}
        bad_setting_bytes = AnprDetectorMessage(sdk_msg).to_bytes()
        good = _detector_message_bytes("obj-2")
        processor = _FakeProcessor([bad_setting_bytes, good])
        ocr_cache = OcrCache(engine=None, output_name=module.DEFAULT_OCR_OUTPUT_NAME)
        event_cache = EventDeduplicationCache(ttl_sec=300.0)

        with self.assertRaises(KeyboardInterrupt):
            module.run_message_loop(processor, ocr_cache, event_cache)

        self.assertEqual(len(processor.sent), 2)


class TestOcrModelContract(unittest.TestCase):
    """The OCR tensor name and shape are looked up by value, so a mismatch is silent.

    A wrong ocr_output_name makes get_binary_output_f32 return None and the OCR
    path exit without a single log line: zero recognitions, zero errors. Pin both
    constants to the shipped model instead.
    """

    model_path = os.path.join(repo_root, "example_models", "License_Plate_OCR.onnx")

    def _graph_output(self):
        onnx = importlib.import_module("onnx")
        if not os.path.exists(self.model_path):
            self.skipTest(f"{self.model_path} not present")
        graph = onnx.load(self.model_path).graph
        self.assertEqual(len(graph.output), 1, "model must expose exactly one output")
        return onnx, graph.output[0]

    def setUp(self):
        try:
            importlib.import_module("onnx")
        except ImportError:
            self.skipTest("onnx not installed")

    def test_output_name_matches_default(self):
        _, output = self._graph_output()
        self.assertEqual(output.name, module.DEFAULT_OCR_OUTPUT_NAME)

    def test_output_shape_matches_expected_logits_shape(self):
        _, output = self._graph_output()
        # Leading dim is a symbolic batch/detection count; the logits are the tail.
        dims = [d.dim_value for d in output.type.tensor_type.shape.dim[1:]]
        self.assertEqual(tuple(dims), module.EXPECTED_LOGITS_SHAPE)

    def test_output_is_float32(self):
        onnx, output = self._graph_output()
        self.assertEqual(output.type.tensor_type.elem_type, onnx.TensorProto.FLOAT)


if __name__ == "__main__":
    unittest.main()
