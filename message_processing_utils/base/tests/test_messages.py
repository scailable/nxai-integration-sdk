"""
Unit tests for general message classes.
"""

import unittest
from datetime import datetime
from unittest.mock import patch
import sys
import os

import msgpack
import numpy as np

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_nxai_utilities = os.path.join(_repo_root, "nxai-utilities")
if _nxai_utilities not in sys.path:
    sys.path.insert(0, _nxai_utilities)

from nxai_postprocessor_sdk import EventMetadata, ObjectsPostProcessorMessage
from message_processing_utils.base import messages as messages_module
from message_processing_utils.base.messages import (
    InferenceMessage,
    GenericMessage,
    parse_typed_message,
)
from message_processing_utils.general.detector import DetectorMessage
from message_processing_utils.anpr.ocr import CctOcrMessage


def _sdk_message(**overrides):
    msg = ObjectsPostProcessorMessage()
    for key, value in overrides.items():
        setattr(msg, key, value)
    return msg


class TestInferenceMessage(unittest.TestCase):
    def test_device_id_property(self):
        uuid_str = "550e8400-e29b-41d4-a716-446655440000"
        msg = GenericMessage(_sdk_message(device_id=uuid_str))
        self.assertEqual(msg.device_id, uuid_str.lower())

    def test_device_id_property_normalizes(self):
        uuid_str = "550e8400-e29b-41d4-a716-446655440000"
        msg = GenericMessage(_sdk_message(device_id=f"  {uuid_str}  "))
        self.assertEqual(msg.device_id, uuid_str.lower())

    def test_device_id_none(self):
        msg = GenericMessage(_sdk_message(device_id=None))
        self.assertIsNone(msg.device_id)

    def test_normalize_device_id_invalid_returns_none(self):
        result = InferenceMessage.normalize_device_id("CAMERA-001")
        self.assertIsNone(result)

    def test_normalize_device_id_uuid(self):
        uuid_str = "550e8400-e29b-41d4-a716-446655440000"
        result = InferenceMessage.normalize_device_id(uuid_str)
        self.assertEqual(uuid_str.lower(), result)

    def test_normalize_device_id_none(self):
        result = InferenceMessage.normalize_device_id(None)
        self.assertIsNone(result)

    def test_normalize_device_id_with_whitespace(self):
        uuid_str = "550e8400-e29b-41d4-a716-446655440000"
        result = InferenceMessage.normalize_device_id(f"  {uuid_str}  ")
        self.assertEqual(result, uuid_str.lower())

    def test_timestamp_small_value_is_still_microseconds(self):
        # The engine always writes Timestamp as timestampUs, so a small value is
        # a timestamp near the epoch, not a second-resolution one.
        msg = GenericMessage(_sdk_message(timestamp=1234567890))
        result = msg.timestamp
        self.assertIsInstance(result, datetime)
        self.assertEqual(result, datetime.fromtimestamp(1234567890 / 1_000_000))

    def test_timestamp_microseconds(self):
        microseconds = 1770135476395000
        msg = GenericMessage(_sdk_message(timestamp=microseconds))
        result = msg.timestamp
        self.assertIsInstance(result, datetime)
        expected = datetime.fromtimestamp(microseconds / 1_000_000)
        self.assertEqual(result, expected)

    def test_timestamp_none(self):
        msg = GenericMessage(_sdk_message(timestamp=None))
        self.assertIsNone(msg.timestamp)

    def test_timestamp_already_datetime(self):
        dt = datetime(2024, 1, 1, 12, 0, 0)
        msg = GenericMessage(_sdk_message(timestamp=dt))
        self.assertEqual(msg.timestamp, dt)

    def test_timestamp_invalid_type(self):
        msg = GenericMessage(_sdk_message(timestamp="invalid-string"))
        with self.assertRaises(TypeError) as context:
            _ = msg.timestamp
        self.assertIn("unexpected type", str(context.exception).lower())
        self.assertIn("Expected int, float, or datetime", str(context.exception))

    def test_user_settings_property(self):
        msg = GenericMessage(_sdk_message(user_settings={"min_confidence": "0.9"}))
        self.assertEqual(msg.user_settings, {"min_confidence": "0.9"})

    def test_user_settings_property_default_empty(self):
        msg = GenericMessage(_sdk_message())
        self.assertEqual(msg.user_settings, {})

    def test_add_event_creates_events_list(self):
        msg = GenericMessage(_sdk_message())
        msg.add_event("test.event", "Test description")
        self.assertEqual(len(msg._msg.events), 1)
        self.assertEqual(msg._msg.events[0].type_id, "test.event")
        self.assertEqual(msg._msg.events[0].description, "Test description")

    def test_add_event_appends_to_existing_list(self):
        existing = EventMetadata()
        existing.type_id = "existing.event"
        existing.description = "Existing desc"
        msg = GenericMessage(_sdk_message(events=[existing]))
        msg.add_event("test.event", "Test description")
        self.assertEqual(len(msg._msg.events), 2)
        self.assertEqual(msg._msg.events[1].type_id, "test.event")

    def test_add_event_structure(self):
        msg = GenericMessage(_sdk_message())
        msg.add_event(
            "speed.measurement",
            "License plate ABC123 detected with average speed 72.0 km/h",
        )
        event = msg._msg.events[0]
        self.assertEqual(event.type_id, "speed.measurement")
        self.assertEqual(
            event.description,
            "License plate ABC123 detected with average speed 72.0 km/h",
        )


class TestUpdateOriginalObject(unittest.TestCase):
    def test_adds_attribute_only_entry_for_the_original_object(self):
        msg = GenericMessage(_sdk_message(original_object_id="obj-1"))

        msg.update_original_object("LP", attributes={"Plate": "ABC123"}, remove_attributes=["Old"])

        root = msgpack.unpackb(msg.to_bytes(), raw=False)
        self.assertEqual(
            root["Objects"]["LP"],
            [{"ID": "obj-1", "attributes": {"Plate": "ABC123"}, "removeAttributes": ["Old"]}],
        )

    def test_repeated_calls_merge_into_one_entry(self):
        msg = GenericMessage(_sdk_message(original_object_id="obj-1"))

        msg.update_original_object("LP", attributes={"A": "1"})
        msg.update_original_object("LP", attributes={"B": "2"}, remove_attributes=["C"])
        msg.update_original_object("LP", remove_attributes=["C"])

        objects = msg._msg.objects["LP"]
        self.assertEqual(len(objects), 1)
        self.assertEqual(objects[0].attributes, {"A": "1", "B": "2"})
        self.assertEqual(objects[0].remove_attributes, ["C"])

    def test_without_original_object_id_does_nothing(self):
        msg = GenericMessage(_sdk_message())

        self.assertIsNone(msg.update_original_object("LP", attributes={"A": "1"}))
        self.assertEqual(msg._msg.objects, {})


class TestParseTypedMessage(unittest.TestCase):
    def test_truncated_bytes_returns_none(self):
        result = parse_typed_message(b"\xff\xff\xff", DetectorMessage)
        self.assertIsNone(result)

    def test_non_map_root_returns_none(self):
        result = parse_typed_message(msgpack.packb([1, 2, 3]), DetectorMessage)
        self.assertIsNone(result)

    def test_scalar_root_returns_none(self):
        result = parse_typed_message(msgpack.packb(42), DetectorMessage)
        self.assertIsNone(result)

    def test_missing_objects_key_still_parses(self):
        data = msgpack.packb({"Timestamp": 1, "DeviceID": "d", "DeviceName": "n"})
        result = parse_typed_message(data, DetectorMessage)
        self.assertIsInstance(result, DetectorMessage)

    def test_original_object_id_without_tensors_routes_to_detector(self):
        sdk_msg = ObjectsPostProcessorMessage()
        data = sdk_msg.to_bytes()
        raw = msgpack.unpackb(data, raw=False)
        raw["OriginalObjectID"] = "parent-1"
        data = msgpack.packb(raw, use_bin_type=True)

        result = parse_typed_message(data, DetectorMessage)
        self.assertIsInstance(result, DetectorMessage)

    def test_original_object_id_with_tensors_routes_to_ocr(self):
        sdk_msg = ObjectsPostProcessorMessage()
        data = sdk_msg.to_bytes()
        raw = msgpack.unpackb(data, raw=False)
        raw["OriginalObjectID"] = "parent-1"
        raw["Tensors"] = [{"Name": "Identity:0", "Data": b"\x00\x00\x00\x00"}]
        data = msgpack.packb(raw, use_bin_type=True)

        result = parse_typed_message(data, DetectorMessage)
        self.assertIsInstance(result, CctOcrMessage)

    def test_no_original_object_id_routes_to_detector_even_with_tensors(self):
        sdk_msg = ObjectsPostProcessorMessage()
        data = sdk_msg.to_bytes()
        raw = msgpack.unpackb(data, raw=False)
        raw["Tensors"] = [{"Name": "Identity:0", "Data": b"\x00\x00\x00\x00"}]
        data = msgpack.packb(raw, use_bin_type=True)

        result = parse_typed_message(data, DetectorMessage)
        self.assertIsInstance(result, DetectorMessage)

    def test_unpacks_bytes_exactly_once(self):
        sdk_msg = ObjectsPostProcessorMessage()
        data = sdk_msg.to_bytes()

        # Patch the msgpack object `messages` itself calls through, not the
        # `sys.modules` entry a string target would resolve. A test module that
        # replaces `sys.modules["msgpack"]` at import scope leaves `messages`
        # holding the real module, so `patch("msgpack.unpackb")` would decorate
        # a different object and count zero calls while the parse still
        # succeeds -- see TestSysModulesIdentity below.
        with patch.object(
            messages_module.msgpack,
            "unpackb",
            wraps=messages_module.msgpack.unpackb,
        ) as mock_unpackb:
            result = parse_typed_message(data, DetectorMessage)

        self.assertIsInstance(result, DetectorMessage)
        self.assertEqual(mock_unpackb.call_count, 1)


class TestSysModulesIdentity(unittest.TestCase):
    """Guard the invariant that string-target patching of `msgpack` relies on.

    A test module that assigns a mock into `sys.modules` at import scope --
    `sys.modules["msgpack"] = MagicMock()`, the shape v4-era example tests used
    to stub absent dependencies -- is never restored, and every later
    `patch("msgpack.<attr>")` in the whole run then patches the mock instead of
    the module the production code holds. The symptom is a passing assertion on
    the parse result next to a mock that was never called, which is why this is
    pinned as its own case rather than left implicit in the test above.
    """

    def test_msgpack_in_sys_modules_is_the_module_the_code_calls(self):
        self.assertIs(sys.modules["msgpack"], messages_module.msgpack)


class TestGetBinaryOutputF32(unittest.TestCase):
    def test_tensor_absent_returns_none(self):
        msg = GenericMessage(_sdk_message(), raw_tensors=[])
        self.assertIsNone(msg.get_binary_output_f32("Identity:0"))

    def test_data_as_list_of_ints_decoded(self):
        values = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tensors = [{"Name": "Identity:0", "Data": list(values.tobytes())}]
        msg = GenericMessage(_sdk_message(), raw_tensors=tensors)
        result = msg.get_binary_output_f32("Identity:0")
        np.testing.assert_array_equal(result, values)

    def test_data_wrong_type_raises_value_error(self):
        tensors = [{"Name": "Identity:0", "Data": 12345}]
        msg = GenericMessage(_sdk_message(), raw_tensors=tensors)
        with self.assertRaises(ValueError):
            msg.get_binary_output_f32("Identity:0")

    def test_decodes_matching_tensor_among_several(self):
        values = np.array([4.0, 5.0], dtype=np.float32)
        tensors = [
            {"Name": "other:0", "Data": np.array([9.0], dtype=np.float32).tobytes()},
            {"Name": "Identity:0", "Data": values.tobytes()},
        ]
        msg = GenericMessage(_sdk_message(), raw_tensors=tensors)
        result = msg.get_binary_output_f32("Identity:0")
        np.testing.assert_array_equal(result, values)


if __name__ == "__main__":
    unittest.main()
