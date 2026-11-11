"""
Unit tests for detection message classes.
"""

import unittest
import sys
import os

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_nxai_utilities = os.path.join(_repo_root, "nxai-utilities")
if _nxai_utilities not in sys.path:
    sys.path.insert(0, _nxai_utilities)

from nxai_postprocessor_sdk import ObjectMetadata, ObjectsPostProcessorMessage
from message_processing_utils.general.detector.messages import DetectorMessage


def _object(track_id):
    obj = ObjectMetadata()
    obj.track_id = track_id
    return obj


class TestDetectorMessage(unittest.TestCase):
    def test_update_metadata_sets_attributes_on_matching_object(self):
        sdk_msg = ObjectsPostProcessorMessage()
        sdk_msg.objects = {"car": [_object("obj-1")]}
        msg = DetectorMessage(sdk_msg)

        success = msg.update_metadata("obj-1", "TEXT", 0.95)

        self.assertTrue(success)
        obj = sdk_msg.objects["car"][0]
        self.assertEqual(obj.attributes["recognized_text"], "TEXT")
        self.assertEqual(obj.attributes["confidence"], "0.95")

    def test_update_metadata_finds_object_across_classes(self):
        sdk_msg = ObjectsPostProcessorMessage()
        sdk_msg.objects = {
            "car": [_object("obj-1")],
            "truck": [_object("obj-2")],
        }
        msg = DetectorMessage(sdk_msg)

        success = msg.update_metadata("obj-2", "TEXT", 0.5)

        self.assertTrue(success)
        self.assertEqual(sdk_msg.objects["truck"][0].attributes["recognized_text"], "TEXT")
        self.assertEqual(sdk_msg.objects["car"][0].attributes, {})

    def test_update_metadata_unknown_object_id_returns_false(self):
        sdk_msg = ObjectsPostProcessorMessage()
        sdk_msg.objects = {"car": [_object("obj-1")]}
        msg = DetectorMessage(sdk_msg)

        self.assertFalse(msg.update_metadata("unknown", "TEXT", 0.5))

    def test_update_metadata_none_object_id_returns_false(self):
        sdk_msg = ObjectsPostProcessorMessage()
        sdk_msg.objects = {"car": [_object(None)]}
        msg = DetectorMessage(sdk_msg)

        self.assertFalse(msg.update_metadata(None, "TEXT", 0.5))


if __name__ == '__main__':
    unittest.main()
