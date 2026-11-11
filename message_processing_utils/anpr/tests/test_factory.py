"""
Unit tests for the ANPR message factories.
"""

import unittest
import sys
import os

import msgpack

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_nxai_utilities = os.path.join(_repo_root, "nxai-utilities")
if _nxai_utilities not in sys.path:
    sys.path.insert(0, _nxai_utilities)

from nxai_postprocessor_sdk import ObjectsPostProcessorMessage
from message_processing_utils.anpr import (
    AnprDetectorMessage,
    SpeedDetectorMessage,
    create_anpr_message_from_bytes,
    create_speed_message_from_bytes,
)
from message_processing_utils.anpr.ocr import CctOcrMessage


def _detector_bytes():
    return ObjectsPostProcessorMessage().to_bytes()


def _ocr_bytes():
    raw = msgpack.unpackb(ObjectsPostProcessorMessage().to_bytes(), raw=False)
    raw["OriginalObjectID"] = "parent-1"
    raw["Tensors"] = [{"Name": "Identity:0", "Data": b"\x00\x00\x00\x00"}]
    return msgpack.packb(raw, use_bin_type=True)


class TestCreateSpeedMessageFromBytes(unittest.TestCase):
    def test_detector_message_is_speed_message(self):
        result = create_speed_message_from_bytes(_detector_bytes())
        self.assertIsInstance(result, SpeedDetectorMessage)

    def test_ocr_message_still_routes_to_ocr(self):
        result = create_speed_message_from_bytes(_ocr_bytes())
        self.assertIsInstance(result, CctOcrMessage)

    def test_unparseable_bytes_return_none(self):
        self.assertIsNone(create_speed_message_from_bytes(b"\xff\xff\xff"))


class TestCreateAnprMessageFromBytes(unittest.TestCase):
    def test_detector_message_is_not_a_speed_message(self):
        result = create_anpr_message_from_bytes(_detector_bytes())
        self.assertIsInstance(result, AnprDetectorMessage)
        self.assertNotIsInstance(result, SpeedDetectorMessage)


if __name__ == "__main__":
    unittest.main()
