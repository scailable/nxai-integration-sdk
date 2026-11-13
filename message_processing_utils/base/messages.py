"""
General message classes for processing inference results from AI Manager.

These classes are usable by all postprocessors, not just detection or OCR.
"""

import logging
import uuid
from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime
import msgpack
import numpy as np
import nxai_postprocessor_sdk


logger = logging.getLogger(__name__)


class InferenceMessage(ABC):
    """Represents a generic inference result message from AI Manager.

    Wraps a typed `nxai_postprocessor_sdk.ObjectsPostProcessorMessage` plus the
    raw `Tensors` array, which the SDK deliberately leaves unparsed (see
    `.claude/docs/input_from_ai_manager.md`) so it's read straight off the
    unpacked msgpack dict here instead.
    """

    def __init__(self, sdk_message, raw_tensors=None):
        self._msg = sdk_message
        self._raw_tensors = raw_tensors or []

    @property
    def original_object_id(self):
        return self._msg.original_object_id

    @property
    def device_id(self) -> Optional[str]:
        """Get normalized DeviceID from message."""
        return self.normalize_device_id(self._msg.device_id)

    @property
    def timestamp(self) -> Optional[datetime]:
        """Get Timestamp (microseconds since the epoch) as a datetime."""
        timestamp_value = self._msg.timestamp
        if timestamp_value is None:
            return None
        if isinstance(timestamp_value, datetime):
            return timestamp_value
        if isinstance(timestamp_value, (int, float)):
            # The v5 engine always writes Timestamp as timestampUs, an i64, so a
            # numeric value is always microseconds.
            try:
                return datetime.fromtimestamp(timestamp_value / 1_000_000)
            except (ValueError, OSError) as e:
                raise ValueError(
                    f"Failed to convert timestamp {timestamp_value} to datetime: {e}"
                ) from e
        raise TypeError(
            f"Timestamp has unexpected type {type(timestamp_value).__name__}: {timestamp_value}. "
            f"Expected int, float, or datetime."
        )

    @property
    def objects(self):
        return self._msg.objects

    @property
    def user_settings(self):
        """Forwarded GUI settings map, read fresh from this message."""
        return self._msg.user_settings

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
    def create_from_bytes(cls, data: bytes) -> Optional["InferenceMessage"]:
        """
        Factory: parse bytes and return a typed message instance.

        Args:
            data: MessagePack-encoded message bytes.

        Returns:
            Typed message instance (DetectorMessage or CctOcrMessage), or None if
            ``data`` could not be parsed.
        """
        from message_processing_utils.general.detector import DetectorMessage

        return parse_typed_message(data, DetectorMessage)

    @abstractmethod
    def handle(self, ocr_cache) -> None:
        """Process the message and mutate object for response."""
        pass

    @property
    def object_ids(self) -> list[str]:
        ids: list[str] = []
        for objs in self._msg.objects.values():
            for obj in objs:
                if obj.track_id is not None:
                    ids.append(obj.track_id)
        return ids

    def find_object(self, object_id: str):
        """Find the ObjectMetadata with the given track_id across all classes."""
        if object_id is None:
            return None
        for objs in self._msg.objects.values():
            for obj in objs:
                if obj.track_id == object_id:
                    return obj
        return None

    def update_original_object(self, class_name: str, attributes=None, remove_attributes=None):
        """Edit the attributes of the object this chained-stage input was cropped from.

        Only meaningful on a message from a model chained in Feature Extraction
        mode, which carries the parent's track ID as ``OriginalObjectID`` but none
        of its other fields. Adds an attribute-only entry for that ID under
        ``class_name`` (merging into one already there), so the engine keeps the
        object's bounding box and confidence and applies just ``attributes``
        (set/overwrite) and ``remove_attributes`` (delete, applied after).

        Returns:
            The ObjectMetadata entry, or None if the message has no OriginalObjectID.
        """
        object_id = self.original_object_id
        if not object_id:
            return None
        class_objects = self._msg.objects.setdefault(class_name, [])
        obj = next((o for o in class_objects if o.track_id == object_id), None)
        if obj is None:
            obj = nxai_postprocessor_sdk.ObjectMetadata.attribute_update(object_id)
            class_objects.append(obj)
        for name, value in (attributes or {}).items():
            obj.attributes[name] = value
        for name in remove_attributes or []:
            if name not in obj.remove_attributes:
                obj.remove_attributes.append(name)
        return obj

    def to_bytes(self):
        """Serialize to MessagePack"""
        return self._msg.to_bytes()

    def get_binary_output_f32(self, name: str):
        for tensor in self._raw_tensors:
            if tensor.get("Name") != name:
                continue
            data = tensor.get("Data")
            if isinstance(data, list):
                data = bytes(data)
            if not isinstance(data, (bytes, bytearray)):
                raise ValueError(f"Tensors[{name}].Data is not bytes")
            return np.frombuffer(data, dtype=np.float32)
        return None

    def add_event(self, event_id: str, description: str) -> None:
        """
        Add an event to the message.

        Args:
            event_id: Unique identifier for the event (e.g., "speed.measurement").
            description: Detailed description of the event.
        """
        event = nxai_postprocessor_sdk.EventMetadata()
        event.type_id = event_id
        event.description = description
        self._msg.events.append(event)


class GenericMessage(InferenceMessage):
    """Fallback for messages that were not classified."""

    def __init__(self, sdk_message, raw_tensors=None):
        super().__init__(sdk_message, raw_tensors)

    def handle(self, ocr_cache) -> None:
        logger.debug("Unknown message type, returning unchanged")
        return None


def parse_typed_message(data: bytes, detector_cls) -> Optional[InferenceMessage]:
    """Parse msgpack bytes into a ``detector_cls`` or ``CctOcrMessage`` instance.

    Shared by ``InferenceMessage.create_from_bytes`` and the ANPR factory
    (``message_processing_utils.anpr.create_anpr_message_from_bytes``): one
    msgpack parse, one classification rule. Returns None (never a message
    wrapping a None SDK message) if ``data`` is not parseable, so callers can
    skip the message instead of crashing on it.

    A message is routed to OCR only when it carries both a chained-stage
    parent linkage (`original_object_id`) and a non-empty raw `Tensors` block
    -- a chained message without tensors is still a detector message.
    """
    from message_processing_utils.anpr.ocr import CctOcrMessage

    try:
        raw_data = msgpack.unpackb(data, raw=False, strict_map_key=False)
    except Exception:
        return None
    if not isinstance(raw_data, dict):
        return None
    raw_tensors = raw_data.get("Tensors", [])

    sdk_msg = nxai_postprocessor_sdk.ObjectsPostProcessorMessage.from_dict(raw_data)
    if sdk_msg is None:
        return None
    if sdk_msg.original_object_id is not None and raw_tensors:
        return CctOcrMessage(sdk_msg, raw_tensors)
    return detector_cls(sdk_msg, raw_tensors)
