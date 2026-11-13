"""
ANPR message processing (detector + OCR) and ANPR factory.
"""

from message_processing_utils.anpr.detector import (
    AnprDetectorMessage,
    SpeedDetectorMessage,
    EventDeduplicationCache,
)
from message_processing_utils.anpr.ocr import CctOcrMessage


def create_anpr_message_from_bytes(data: bytes):
    """
    Factory: parse bytes and return an ANPR-compatible message instance, or
    None if ``data`` could not be parsed.
    For detector messages returns AnprDetectorMessage so that handle() writes
    "License Plate Text" and "Confidence" attributes expected by the UI.
    """
    from message_processing_utils.base.messages import parse_typed_message

    return parse_typed_message(data, AnprDetectorMessage)


def create_speed_message_from_bytes(data: bytes):
    """
    Factory: like ``create_anpr_message_from_bytes``, but returns detector
    messages as SpeedDetectorMessage so speed metadata can be attached on top
    of the license-plate attributes. Returns None if ``data`` could not be
    parsed.
    """
    from message_processing_utils.base.messages import parse_typed_message

    return parse_typed_message(data, SpeedDetectorMessage)


__all__ = [
    "AnprDetectorMessage",
    "SpeedDetectorMessage",
    "EventDeduplicationCache",
    "CctOcrMessage",
    "create_anpr_message_from_bytes",
    "create_speed_message_from_bytes",
]
