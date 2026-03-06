"""
ANPR message processing (detector + OCR) and ANPR factory.
"""

from message_processing_utils.anpr.detector import AnprDetectorMessage, SpeedDetectorMessage
from message_processing_utils.anpr.ocr import CctOcrMessage


def create_anpr_message_from_bytes(data: bytes):
    """
    Factory: parse bytes and return an ANPR-compatible message instance.
    For detector messages returns AnprDetectorMessage so that handle() writes
    "License Plate Text" and "Confidence" attributes expected by the UI.
    """
    from message_processing_utils.base.messages import (
        InferenceMessage,
        GenericMessage,
    )

    msg = InferenceMessage.from_payload(data)
    is_ocr = msg.get("OriginalObjectID") is not None and bool(
        msg.get("BinaryOutputs", [])
    )
    # Treat as detector if key present (even when BBoxes_xyxy is {}), so response format stays consistent
    is_detector = "BBoxes_xyxy" in msg
    if is_detector:
        return AnprDetectorMessage(msg)
    if is_ocr:
        return CctOcrMessage(msg)
    return GenericMessage(msg)


__all__ = [
    "AnprDetectorMessage",
    "SpeedDetectorMessage",
    "CctOcrMessage",
    "create_anpr_message_from_bytes",
]
