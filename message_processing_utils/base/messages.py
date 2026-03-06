"""
General message classes for processing inference results from AI Manager.

These classes are usable by all postprocessors, not just detection or OCR.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from typing import Optional
from pprint import pformat
from datetime import datetime
import msgpack
import numpy as np


logger = logging.getLogger(__name__)


class InferenceMessage(ABC):
    """Represents a generic inference result message from AI Manager."""

    def __init__(self, message: dict):
        self._message = message or {}

    @property
    def original_object_id(self):
        return self._message.get("OriginalObjectID")

    @original_object_id.setter
    def original_object_id(self, value):
        self._message["OriginalObjectID"] = value

    @property
    def device_id(self) -> Optional[str]:
        """Get normalized DeviceID from message."""
        return self.normalize_device_id(self._message.get("DeviceID"))

    @property
    def timestamp(self) -> Optional[datetime]:
        """Get Timestamp from message, converted to datetime."""
        timestamp_value = self._message.get("Timestamp")
        if timestamp_value is None:
            return None
        if isinstance(timestamp_value, datetime):
            return timestamp_value
        if isinstance(timestamp_value, (int, float)):
            # Check if timestamp is in microseconds (value > 1e12)
            # Unix timestamp in seconds is typically < 1e10 (year 2286)
            # Timestamps > 1e12 are likely in microseconds
            if timestamp_value > 1e12:
                # Convert microseconds to seconds
                timestamp_value = timestamp_value / 1_000_000
            try:
                return datetime.fromtimestamp(timestamp_value)
            except (ValueError, OSError) as e:
                raise ValueError(
                    f"Failed to convert timestamp {timestamp_value} to datetime: {e}"
                ) from e
        raise TypeError(
            f"Timestamp has unexpected type {type(timestamp_value).__name__}: {timestamp_value}. "
            f"Expected int, float, or datetime."
        )

    @property
    def _binary_outputs(self):
        if "BinaryOutputs" not in self._message or self._message["BinaryOutputs"] is None:
            self._message["BinaryOutputs"] = []
        return self._message["BinaryOutputs"]

    @property
    def bboxes_xyxy(self):
        if "BBoxes_xyxy" not in self._message or self._message["BBoxes_xyxy"] is None:
            self._message["BBoxes_xyxy"] = {}
        return self._message["BBoxes_xyxy"]

    @property
    def objects_metadata(self):
        if "ObjectsMetaData" not in self._message or self._message["ObjectsMetaData"] is None:
            self._message["ObjectsMetaData"] = {}
        return self._message["ObjectsMetaData"]

    @property
    def inference_data(self):
        return self._message.get("InferenceData")

    @staticmethod
    def normalize_device_id(device_id) -> Optional[str]:
        """Normalize UUID to string for comparison (canonical form via uuid.UUID)."""
        if device_id is None:
            logger.error("normalize_device_id: device_id is None")
            return None
        s = str(device_id).strip()
        if not s:
            logger.error("normalize_device_id: empty device_id")
            return None
        try:
            return str(uuid.UUID(s))
        except (ValueError, TypeError) as e:
            logger.error("normalize_device_id: invalid UUID %r: %s", device_id, e)
            return None

    @classmethod
    def from_payload(cls, data):
        try:
            msg = msgpack.unpackb(data, raw=False, strict_map_key=False)
        except Exception as e:
            logger.error("Failed to unpack message: %s", e)
            raise
        logger.debug("Received message: %s", pformat(msg))
        return msg

    @classmethod
    def create_from_bytes(cls, data: bytes) -> "InferenceMessage":
        """
        Factory: parse bytes and return a typed message instance.

        Args:
            data: MessagePack-encoded message bytes.

        Returns:
            Typed message instance (DetectorMessage, CctOcrMessage, or GenericMessage).
        """
        from message_processing_utils.general.detector import DetectorMessage
        from message_processing_utils.anpr.ocr import CctOcrMessage

        msg = cls.from_payload(data)
        is_ocr = msg.get("OriginalObjectID") is not None and bool(
            msg.get("BinaryOutputs", [])
        )
        is_detector = "BBoxes_xyxy" in msg
        if is_detector:
            return DetectorMessage(msg)
        if is_ocr:
            return CctOcrMessage(msg)
        return GenericMessage(msg)

    @abstractmethod
    def handle(self, ocr_cache) -> None:
        """Process the message and mutate object for response."""
        pass

    @property
    def object_ids(self) -> list[str]:
        ids: list[str] = []
        for meta in (self.objects_metadata or {}).values():
            ids.extend(meta.get("ObjectIDs", []))
        return ids

    def to_bytes(self):
        """Serialize to MessagePack"""
        logger.debug("Sending message: %s", pformat(self._message))
        return msgpack.packb(self._message, use_bin_type=True)

    def _get_binary_output(self, name: str):
        for output in self._binary_outputs:
            if output.get("Name") == name:
                return output
        return None

    def get_binary_output_f32(self, name: str):
        output = self._get_binary_output(name)
        if not output:
            return None
        data = output.get("Data")
        if isinstance(data, list):
            data = bytes(data)
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError(f"BinaryOutputs[{name}].Data is not bytes")
        return np.frombuffer(data, dtype=np.float32)

    def add_event(self, event_id: str, caption: str, description: str) -> None:
        """
        Add an event to the message.

        Args:
            event_id: Unique identifier for the event (e.g., "speed.measurement").
            caption: Short title for the event.
            description: Detailed description of the event.
        """
        if "Events" not in self._message:
            self._message["Events"] = []
        self._message["Events"].append({
            "ID": event_id,
            "Caption": caption,
            "Description": description,
        })


class GenericMessage(InferenceMessage):
    """Fallback for messages that were not classified."""

    def __init__(self, message: dict):
        super().__init__(message)

    def handle(self, ocr_cache) -> None:
        logger.debug("Unknown message type, returning unchanged")
        return None
