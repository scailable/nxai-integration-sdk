"""
Unit tests for detection message classes.
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Mock msgpack only (DetectorMessage uses it in to_bytes); numpy is not used here
sys.modules['msgpack'] = MagicMock()

# Add repo root to path so message_processing_utils is importable
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from message_processing_utils.general.detector.messages import DetectorMessage


class TestDetectorMessage(unittest.TestCase):
    def test_update_metadata_new_structure(self):
        """Test creating ObjectsMetaData from scratch."""
        msg = DetectorMessage({
            "BBoxes_xyxy": {"car": [0, 0, 1, 1]},
            "ObjectsMetaData": {"car": {"ObjectIDs": ["obj-1"]}}
        })
        success = msg.update_metadata("obj-1", "TEXT", 0.95)
        self.assertTrue(success)
        meta = msg.objects_metadata["car"]
        self.assertEqual(
            meta["AttributeValues"][0][meta["AttributeKeys"][0].index("recognized_text")],
            "TEXT"
        )
        self.assertEqual(
            meta["AttributeValues"][0][meta["AttributeKeys"][0].index("confidence")],
            "0.95"
        )

    def test_multiple_bboxes_normalization(self):
        """Test normalization when some boxes have IDs and some don't."""
        msg_dict = {
            "BBoxes_xyxy": {"car": [0, 0, 1, 1, 2, 2, 3, 3]},  # 2 boxes
            "ObjectsMetaData": {"car": {"ObjectIDs": ["id-1"]}}  # Only 1 ID
        }
        msg = DetectorMessage(msg_dict)
        # to_bytes should normalize ObjectIDs to length 2
        with patch('message_processing_utils.general.detector.messages.msgpack.packb', side_effect=lambda x, **kwargs: x):
            payload = msg.to_bytes()
            self.assertEqual(len(payload["ObjectsMetaData"]["car"]["ObjectIDs"]), 2)
            self.assertIsNone(payload["ObjectsMetaData"]["car"]["ObjectIDs"][1])


if __name__ == '__main__':
    unittest.main()
