"""
Unit tests for general message classes.
"""

import unittest
from unittest.mock import MagicMock
from datetime import datetime
import sys
import os

sys.modules["msgpack"] = MagicMock()

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from message_processing_utils.base.messages import InferenceMessage, GenericMessage


class TestInferenceMessage(unittest.TestCase):
    def test_device_id_property(self):
        uuid_str = "550e8400-e29b-41d4-a716-446655440000"
        msg = GenericMessage({"DeviceID": uuid_str})
        self.assertEqual(msg.device_id, uuid_str.lower())

    def test_device_id_property_normalizes(self):
        uuid_str = "550e8400-e29b-41d4-a716-446655440000"
        msg = GenericMessage({"DeviceID": f"  {uuid_str}  "})
        self.assertEqual(msg.device_id, uuid_str.lower())

    def test_device_id_none(self):
        msg = GenericMessage({})
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

    def test_timestamp_property(self):
        msg = GenericMessage({"Timestamp": 1234567890})
        result = msg.timestamp
        self.assertIsInstance(result, datetime)
        self.assertEqual(result, datetime.fromtimestamp(1234567890))

    def test_timestamp_microseconds(self):
        microseconds = 1770135476395000
        msg = GenericMessage({"Timestamp": microseconds})
        result = msg.timestamp
        self.assertIsInstance(result, datetime)
        expected = datetime.fromtimestamp(microseconds / 1_000_000)
        self.assertEqual(result, expected)

    def test_timestamp_none(self):
        msg = GenericMessage({})
        self.assertIsNone(msg.timestamp)

    def test_timestamp_already_datetime(self):
        dt = datetime(2024, 1, 1, 12, 0, 0)
        msg = GenericMessage({"Timestamp": dt})
        self.assertEqual(msg.timestamp, dt)

    def test_timestamp_invalid_type(self):
        msg = GenericMessage({"Timestamp": "invalid-string"})
        with self.assertRaises(TypeError) as context:
            _ = msg.timestamp
        self.assertIn("unexpected type", str(context.exception).lower())
        self.assertIn("Expected int, float, or datetime", str(context.exception))

    def test_add_event_creates_events_list(self):
        msg = GenericMessage({})
        msg.add_event("test.event", "Test Event", "Test description")
        self.assertIn("Events", msg._message)
        self.assertEqual(len(msg._message["Events"]), 1)

    def test_add_event_appends_to_existing_list(self):
        msg = GenericMessage({"Events": [{"ID": "existing.event", "Caption": "Existing", "Description": "Existing desc"}]})
        msg.add_event("test.event", "Test Event", "Test description")
        self.assertEqual(len(msg._message["Events"]), 2)
        self.assertEqual(msg._message["Events"][1]["ID"], "test.event")

    def test_add_event_structure(self):
        msg = GenericMessage({})
        msg.add_event("speed.measurement", "Speed Measurement", "License plate ABC123 detected with average speed 72.0 km/h")
        event = msg._message["Events"][0]
        self.assertEqual(event["ID"], "speed.measurement")
        self.assertEqual(event["Caption"], "Speed Measurement")
        self.assertEqual(event["Description"], "License plate ABC123 detected with average speed 72.0 km/h")


if __name__ == "__main__":
    unittest.main()
