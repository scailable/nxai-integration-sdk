"""
Unit tests for postprocessor-python-measure-average-car-speed (v5 SDK).
"""

import unittest
from unittest.mock import patch
import json
import re
import sys
import os
import importlib.util
import msgpack
from datetime import datetime, timedelta

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.join(script_dir, "..")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
_nxai_utilities = os.path.join(repo_root, "nxai-utilities")
if _nxai_utilities not in sys.path:
    sys.path.insert(0, _nxai_utilities)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from nxai_postprocessor_sdk import ObjectMetadata, ObjectsPostProcessorMessage
from message_processing_utils.anpr import CctOcrMessage, SpeedDetectorMessage
from message_processing_utils.general.ocr import LogitsOcrEngine, OcrCache
from speed_cache import SpeedMeasurementCache

spec = importlib.util.spec_from_file_location(
    "postprocessor_python_measure_average_car_speed",
    os.path.join(script_dir, "postprocessor-python-measure-average-car-speed.py"),
)
module = importlib.util.module_from_spec(spec)
sys.modules["postprocessor_python_measure_average_car_speed"] = module
spec.loader.exec_module(module)

CAMERA_1_ID = "3645c7ee-ca91-e579-e753-1d85af1fd08c"
CAMERA_2_ID = "e3e9a385-7fe0-3ba5-5482-a86cde7faf48"

# The plugin correlates a response to its request on DeviceID + "|" + Timestamp,
# so the fixture must carry a non-zero Timestamp -- the SDK default is 0, which
# would let the echo assertion pass vacuously. CAMERA_1_ID already stands in for
# a non-empty DeviceID.
ECHOED_TIMESTAMP = 1726000000000000


def _object(track_id, license_plate=None):
    obj = ObjectMetadata()
    obj.track_id = track_id
    if license_plate is not None:
        obj.attributes["License Plate Text"] = license_plate
    return obj


def _detector_message(objects, device_id=CAMERA_1_ID, timestamp=None, user_settings=None):
    sdk_msg = ObjectsPostProcessorMessage()
    sdk_msg.objects = {"lp": list(objects)}
    sdk_msg.device_id = device_id
    sdk_msg.timestamp = timestamp
    sdk_msg.user_settings = user_settings or {}
    return SpeedDetectorMessage(sdk_msg)


def _microseconds(moment):
    return int(moment.timestamp() * 1_000_000)


def _camera_config(**overrides):
    config = {
        "camera_1_id": CAMERA_1_ID,
        "camera_2_id": CAMERA_2_ID,
        "distance_between_cameras_m": 100.0,
        "second_appearance_timeout_sec": 60.0,
    }
    config.update(overrides)
    return config


class TestReadCameraConfig(unittest.TestCase):
    def test_defaults_when_nothing_configured(self):
        config = module.read_camera_config({})
        self.assertEqual(config["camera_1_id"], "")
        self.assertEqual(config["camera_2_id"], "")
        self.assertEqual(
            config["distance_between_cameras_m"], module.DEFAULT_DISTANCE_BETWEEN_CAMERAS_M
        )
        self.assertEqual(
            config["second_appearance_timeout_sec"],
            module.DEFAULT_SECOND_APPEARANCE_TIMEOUT_SEC,
        )

    def test_reads_forwarded_user_settings(self):
        config = module.read_camera_config(
            {
                "camera_1_id": CAMERA_1_ID,
                "camera_2_id": CAMERA_2_ID,
                "distance_between_cameras_m": "150.5",
                "second_appearance_timeout_sec": "90",
            }
        )
        self.assertEqual(config["camera_1_id"], CAMERA_1_ID)
        self.assertEqual(config["camera_2_id"], CAMERA_2_ID)
        self.assertEqual(config["distance_between_cameras_m"], 150.5)
        self.assertEqual(config["second_appearance_timeout_sec"], 90.0)

    def test_strips_whitespace_around_camera_ids(self):
        config = module.read_camera_config({"camera_1_id": f"  {CAMERA_1_ID}  "})
        self.assertEqual(config["camera_1_id"], CAMERA_1_ID)

    def test_non_numeric_distance_falls_back_to_default(self):
        config = module.read_camera_config({"distance_between_cameras_m": "not-a-number"})
        self.assertEqual(
            config["distance_between_cameras_m"], module.DEFAULT_DISTANCE_BETWEEN_CAMERAS_M
        )

    def test_non_numeric_timeout_falls_back_to_default(self):
        config = module.read_camera_config({"second_appearance_timeout_sec": "not-a-number"})
        self.assertEqual(
            config["second_appearance_timeout_sec"],
            module.DEFAULT_SECOND_APPEARANCE_TIMEOUT_SEC,
        )


class TestGetFloatSetting(unittest.TestCase):
    def test_valid_value_is_parsed(self):
        self.assertEqual(module._get_float_setting({"x": "12.5"}, "x", 1.0), 12.5)

    def test_missing_key_uses_default(self):
        self.assertEqual(module._get_float_setting({}, "x", 1.0), 1.0)

    def test_invalid_value_uses_default(self):
        self.assertEqual(module._get_float_setting({"x": "abc"}, "x", 1.0), 1.0)

    def test_none_value_uses_default(self):
        self.assertEqual(module._get_float_setting({"x": None}, "x", 1.0), 1.0)


class TestExtractLicensePlates(unittest.TestCase):
    def test_returns_object_id_and_plate(self):
        message = _detector_message([_object("obj-1", "ABC123")])
        self.assertEqual(module._extract_license_plates(message), [("obj-1", "ABC123")])

    def test_strips_surrounding_whitespace(self):
        message = _detector_message([_object("obj-1", "  ABC123 ")])
        self.assertEqual(module._extract_license_plates(message), [("obj-1", "ABC123")])

    def test_multiple_objects(self):
        message = _detector_message(
            [_object("obj-1", "ABC123"), _object("obj-2", "XYZ789")]
        )
        self.assertEqual(
            module._extract_license_plates(message),
            [("obj-1", "ABC123"), ("obj-2", "XYZ789")],
        )

    def test_skips_object_without_track_id(self):
        message = _detector_message([_object(None, "ABC123")])
        self.assertEqual(module._extract_license_plates(message), [])

    def test_skips_object_without_recognized_text(self):
        message = _detector_message([_object("obj-1")])
        self.assertEqual(module._extract_license_plates(message), [])

    def test_skips_blank_recognized_text(self):
        message = _detector_message([_object("obj-1", "   ")])
        self.assertEqual(module._extract_license_plates(message), [])


class TestHandleMessage(unittest.TestCase):
    def setUp(self):
        self.ocr_cache = OcrCache(engine=None, output_name=module.DEFAULT_OCR_OUTPUT_NAME)
        self.speed_cache = SpeedMeasurementCache()
        self.speed_cache.update_config(CAMERA_1_ID, CAMERA_2_ID, 100.0, 60.0)
        self.addCleanup(self.speed_cache.stop)
        self.first_seen = datetime.now()

    def _handle(self, message, camera_config=None):
        return module.handle_message(
            message, self.ocr_cache, self.speed_cache, camera_config or _camera_config()
        )

    def test_ocr_message_is_decoded_but_not_speed_tracked(self):
        import numpy as np

        sdk_msg = ObjectsPostProcessorMessage()
        sdk_msg.original_object_id = "obj-1"
        logits = np.zeros(module.EXPECTED_LOGITS_SHAPE, dtype=np.float32)
        logits[0, 10] = 1.0  # 'A'
        message = CctOcrMessage(
            sdk_msg, [{"Name": module.DEFAULT_OCR_OUTPUT_NAME, "Data": logits.tobytes()}]
        )
        self.ocr_cache = OcrCache(
            LogitsOcrEngine(
                expected_logits_shape=module.EXPECTED_LOGITS_SHAPE,
                char_map=module.CHAR_MAP,
            ),
            module.DEFAULT_OCR_OUTPUT_NAME,
        )

        result = self._handle(message)

        self.assertIs(result, message)
        self.assertEqual(sdk_msg.events, [])
        cached = self.ocr_cache.get_cached_result("obj-1")
        self.assertIsNotNone(cached)
        self.assertTrue(cached[0].startswith("A"))

    def test_single_camera_sighting_emits_nothing(self):
        message = _detector_message(
            [_object("obj-1", "ABC123")],
            device_id=CAMERA_1_ID,
            timestamp=_microseconds(self.first_seen),
        )

        self._handle(message)

        self.assertEqual(message._msg.events, [])
        self.assertNotIn("Speed", message._msg.objects["lp"][0].attributes)

    def test_second_camera_sighting_emits_event_and_speed_attribute(self):
        first = _detector_message(
            [_object("obj-1", "ABC123")],
            device_id=CAMERA_1_ID,
            timestamp=_microseconds(self.first_seen),
        )
        self._handle(first)

        # 100 m in 5 s == 20 m/s == 72.0 km/h.
        second = _detector_message(
            [_object("obj-2", "ABC123")],
            device_id=CAMERA_2_ID,
            timestamp=_microseconds(self.first_seen + timedelta(seconds=5)),
        )
        self._handle(second)

        self.assertEqual(len(second._msg.events), 1)
        event = second._msg.events[0]
        self.assertEqual(event.type_id, module.EVENT_ID_SPEED_MEASUREMENT)
        self.assertIn("ABC123", event.description)
        self.assertIn("72.0 km/h", event.description)
        self.assertEqual(second._msg.objects["lp"][0].attributes["Speed"], "72.0")

    def test_later_frame_reattaches_cached_speed_without_new_event(self):
        self._handle(
            _detector_message(
                [_object("obj-1", "ABC123")],
                device_id=CAMERA_1_ID,
                timestamp=_microseconds(self.first_seen),
            )
        )
        self._handle(
            _detector_message(
                [_object("obj-2", "ABC123")],
                device_id=CAMERA_2_ID,
                timestamp=_microseconds(self.first_seen + timedelta(seconds=5)),
            )
        )

        later = _detector_message(
            [_object("obj-3", "ABC123")],
            device_id=CAMERA_2_ID,
            timestamp=_microseconds(self.first_seen + timedelta(seconds=6)),
        )
        self._handle(later)

        self.assertEqual(later._msg.events, [])
        self.assertEqual(later._msg.objects["lp"][0].attributes["Speed"], "72.0")

    def test_unconfigured_cameras_disable_speed_tracking(self):
        first = _detector_message(
            [_object("obj-1", "ABC123")],
            device_id=CAMERA_1_ID,
            timestamp=_microseconds(self.first_seen),
        )
        self._handle(first, _camera_config(camera_1_id="", camera_2_id=""))

        second = _detector_message(
            [_object("obj-2", "ABC123")],
            device_id=CAMERA_2_ID,
            timestamp=_microseconds(self.first_seen + timedelta(seconds=5)),
        )
        self._handle(second, _camera_config(camera_1_id="", camera_2_id=""))

        self.assertEqual(second._msg.events, [])

    def test_non_positive_distance_disables_speed_tracking(self):
        message = _detector_message(
            [_object("obj-1", "ABC123")],
            device_id=CAMERA_1_ID,
            timestamp=_microseconds(self.first_seen),
        )
        self._handle(message, _camera_config(distance_between_cameras_m=0.0))

        self.assertEqual(message._msg.events, [])

    def test_non_positive_timeout_disables_speed_tracking(self):
        message = _detector_message(
            [_object("obj-1", "ABC123")],
            device_id=CAMERA_1_ID,
            timestamp=_microseconds(self.first_seen),
        )
        self._handle(message, _camera_config(second_appearance_timeout_sec=0.0))

        self.assertEqual(message._msg.events, [])

    def test_message_without_timestamp_is_skipped(self):
        message = _detector_message(
            [_object("obj-1", "ABC123")], device_id=CAMERA_1_ID, timestamp=None
        )

        self._handle(message)

        self.assertEqual(message._msg.events, [])

    def test_sighting_on_unconfigured_camera_is_ignored(self):
        other_camera = "00000000-0000-0000-0000-000000000000"
        self._handle(
            _detector_message(
                [_object("obj-1", "ABC123")],
                device_id=other_camera,
                timestamp=_microseconds(self.first_seen),
            )
        )
        second = _detector_message(
            [_object("obj-2", "ABC123")],
            device_id=CAMERA_2_ID,
            timestamp=_microseconds(self.first_seen + timedelta(seconds=5)),
        )
        self._handle(second)

        self.assertEqual(second._msg.events, [])

    def test_cached_ocr_result_becomes_a_tracked_plate(self):
        class _FakeOcrMessage:
            original_object_id = "obj-1"

        self.ocr_cache.cache_ocr_result(_FakeOcrMessage(), "ABC123", 0.97)

        message = _detector_message(
            [_object("obj-1")],
            device_id=CAMERA_1_ID,
            timestamp=_microseconds(self.first_seen),
        )
        self._handle(message)

        obj = message._msg.objects["lp"][0]
        self.assertEqual(obj.attributes["License Plate Text"], "ABC123")
        self.assertEqual(self.speed_cache.get_speed("ABC123"), None)

        second = _detector_message(
            [_object("obj-2", "ABC123")],
            device_id=CAMERA_2_ID,
            timestamp=_microseconds(self.first_seen + timedelta(seconds=5)),
        )
        self._handle(second)
        self.assertEqual(len(second._msg.events), 1)


class TestResponseEchoesCorrelationFields(unittest.TestCase):
    def setUp(self):
        self.ocr_cache = OcrCache(engine=None, output_name=module.DEFAULT_OCR_OUTPUT_NAME)
        self.speed_cache = SpeedMeasurementCache()
        self.speed_cache.update_config(CAMERA_1_ID, CAMERA_2_ID, 100.0, 60.0)
        self.addCleanup(self.speed_cache.stop)

    def test_serialized_response_carries_the_requests_device_id_and_timestamp(self):
        # The AI Manager routes a response back to the camera that asked for it by
        # the key DeviceID + "|" + Timestamp. A
        # response that does not echo both fields back unchanged is dropped
        # silently -- no log line anywhere -- so nothing but this assertion
        # catches a processor that answers with a freshly built message.
        # The comparison is against the serialized frame, which is what the
        # plugin actually reads.
        message = _detector_message(
            [_object("obj-1", "ABC123")],
            device_id=CAMERA_1_ID,
            timestamp=ECHOED_TIMESTAMP,
        )

        result = module.handle_message(
            message, self.ocr_cache, self.speed_cache, _camera_config()
        )

        serialized = msgpack.unpackb(result.to_bytes(), raw=False)
        self.assertEqual(CAMERA_1_ID, serialized["DeviceID"])
        self.assertEqual(ECHOED_TIMESTAMP, serialized["Timestamp"])


class TestRegistrationJson(unittest.TestCase):
    def test_declares_receive_binary_data(self):
        with open(
            os.path.join(script_dir, "external_postprocessors.json"), encoding="utf-8"
        ) as handle:
            entries = json.load(handle)

        self.assertTrue(
            entries[0].get("ReceiveBinaryData") is True,
            "external_postprocessors.json must set ReceiveBinaryData: true — "
            "the OCR path reads Tensors (base/messages.py get_binary_output_f32)",
        )

    def test_emits_the_event_id_exactly_as_declared(self):
        """The mediaserver drops an emitted id that differs from Events[].ID."""
        with open(
            os.path.join(script_dir, "external_postprocessors.json"), encoding="utf-8"
        ) as handle:
            entry = json.load(handle)[0]

        declared = [event["ID"] for event in entry["Events"]]
        self.assertEqual(declared, ["speed.measurement"])
        self.assertEqual(module.EVENT_ID_SPEED_MEASUREMENT, declared[0])

    def test_declares_every_user_setting_the_processor_reads(self):
        with open(
            os.path.join(script_dir, "external_postprocessors.json"), encoding="utf-8"
        ) as handle:
            entry = json.load(handle)[0]

        declared = {setting["name"] for setting in entry["UserSettings"]}
        self.assertEqual(declared, set(module.read_camera_config({}).keys()))

    def test_readme_snippet_matches_registration_json(self):
        """The README snippet is what operators copy — it must not drift."""
        with open(
            os.path.join(script_dir, "external_postprocessors.json"), encoding="utf-8"
        ) as handle:
            entry = json.load(handle)[0]
        with open(os.path.join(script_dir, "README.md"), encoding="utf-8") as handle:
            readme = handle.read()

        blocks = re.findall(r"```json\n(.*?)```", readme, re.DOTALL)
        self.assertEqual(len(blocks), 1, "README must carry exactly one json snippet")

        self.assertEqual(json.loads(blocks[0]), entry)


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

    def setUp(self):
        self.ocr_cache = OcrCache(engine=None, output_name=module.DEFAULT_OCR_OUTPUT_NAME)
        self.speed_cache = SpeedMeasurementCache()
        self.speed_cache.update_config(CAMERA_1_ID, CAMERA_2_ID, 100.0, 60.0)
        self.addCleanup(self.speed_cache.stop)
        self.good = _detector_message(
            [_object("obj-1", "ABC123")],
            timestamp=_microseconds(datetime.now()),
        ).to_bytes()

    def _run(self, processor):
        with self.assertRaises(KeyboardInterrupt):
            module.run_message_loop(processor, self.ocr_cache, self.speed_cache)

    def test_unparsable_message_does_not_end_the_loop(self):
        processor = _FakeProcessor([b"\xc1not-msgpack", self.good])
        self._run(processor)
        self.assertEqual(
            len(processor.sent), 1, "the message after a bad one must still be served"
        )

    def test_raising_message_does_not_end_the_loop(self):
        processor = _FakeProcessor([RuntimeError("transport hiccup"), self.good])
        self._run(processor)
        self.assertEqual(len(processor.sent), 1)

    def test_timeout_does_not_end_the_loop(self):
        processor = _FakeProcessor([None, None, self.good])
        self._run(processor)
        self.assertEqual(len(processor.sent), 1)

    def test_exit_signal_ends_the_loop_cleanly(self):
        # A graceful EXIT must return, not be swallowed as a plain timeout
        # that keeps the loop spinning until the AI Manager kills the process.
        processor = _FakeProcessor([EXIT, self.good])
        module.run_message_loop(processor, self.ocr_cache, self.speed_cache)
        self.assertEqual(
            len(processor.sent), 0, "loop must stop before serving the next message"
        )

    def test_parsed_message_still_gets_a_response_when_handling_fails(self):
        # A frame that was successfully parsed must not be silently dropped:
        # the engine still needs a response for it, even if handling blew up.
        processor = _FakeProcessor([self.good])
        with patch.object(module, "handle_message", side_effect=RuntimeError("boom")):
            with self.assertRaises(KeyboardInterrupt):
                module.run_message_loop(processor, self.ocr_cache, self.speed_cache)
        self.assertEqual(len(processor.sent), 1)

    def test_failed_send_ends_the_loop(self):
        processor = _FakeProcessor([self.good, self.good], send_results=[False])
        # Returns instead of raising: the script's second message is never read.
        module.run_message_loop(processor, self.ocr_cache, self.speed_cache)
        self.assertEqual(len(processor.sent), 1)


if __name__ == "__main__":
    unittest.main()
