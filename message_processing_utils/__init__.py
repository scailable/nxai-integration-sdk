"""
Message processing utilities for inference results (detection, OCR, ANPR).

For base message types: from message_processing_utils.base import ...
For general detector: from message_processing_utils.general.detector import ...
For general OCR: from message_processing_utils.general.ocr import ...
For ANPR: from message_processing_utils.anpr import ...
"""

from message_processing_utils.base import InferenceMessage, GenericMessage
from message_processing_utils.anpr import (
    create_anpr_message_from_bytes,
    create_speed_message_from_bytes,
)

create_from_bytes = InferenceMessage.create_from_bytes

__all__ = [
    "InferenceMessage",
    "GenericMessage",
    "create_from_bytes",
    "create_anpr_message_from_bytes",
    "create_speed_message_from_bytes",
]
