"""
ANPR detector message classes (license plate, speed).
"""

import logging
from message_processing_utils.general.detector.messages import DetectorMessage


logger = logging.getLogger(__name__)


class AnprDetectorMessage(DetectorMessage):
    """ANPR-specific detector message with custom attribute names."""

    def __init__(self, message: dict):
        super().__init__(message)

    def handle(self, ocr_cache) -> None:
        for object_id in set(self.object_ids):
            cached = ocr_cache.get_cached_result(object_id)
            if cached:
                text, conf = cached
                self.add_license_plate_metadata(object_id, text, conf)
        logger.debug("Returning detector message with cached OCR results")
        return None

    def add_license_plate_metadata(self, object_id: str, recognized_text: str, confidence: float) -> bool:
        if object_id is None:
            return False
        meta_map = self.objects_metadata
        for class_name, meta in meta_map.items():
            ids = meta.get("ObjectIDs", [])
            if object_id not in ids:
                continue
            idx = ids.index(object_id)
            keys = meta.setdefault("AttributeKeys", [])
            vals = meta.setdefault("AttributeValues", [])
            while len(keys) <= idx:
                keys.append([])
            while len(vals) <= idx:
                vals.append([])
            if not isinstance(keys[idx], list):
                keys[idx] = list(keys[idx]) if keys[idx] else []
            if not isinstance(vals[idx], list):
                vals[idx] = list(vals[idx]) if vals[idx] else []
            # Update License Plate Text
            if "License Plate Text" in keys[idx]:
                k_idx = keys[idx].index("License Plate Text")
                if len(vals[idx]) > k_idx:
                    vals[idx][k_idx] = recognized_text
                else:
                    vals[idx].append(recognized_text)
            else:
                keys[idx].append("License Plate Text")
                vals[idx].append(recognized_text)
            # Update Confidence (formatted to 4 decimal places)
            confidence_str = f"{float(confidence):.4f}"
            if "Confidence" in keys[idx]:
                c_idx = keys[idx].index("Confidence")
                if len(vals[idx]) > c_idx:
                    vals[idx][c_idx] = confidence_str
                else:
                    vals[idx].append(confidence_str)
            else:
                keys[idx].append("Confidence")
                vals[idx].append(confidence_str)
            return True
        return False


class SpeedDetectorMessage(AnprDetectorMessage):
    """Speed measurement detector message that extends ANPR functionality with speed metadata."""

    def __init__(self, message: dict):
        super().__init__(message)

    def add_speed_metadata(self, object_id: str, speed_ms: float) -> bool:
        """
        Add speed attribute to object metadata.

        Args:
            object_id: Object ID to add speed attribute to.
            speed_ms: Speed in m/s.

        Returns:
            True if speed was added, False otherwise.
        """
        if object_id is None:
            return False
        meta_map = self.objects_metadata
        for class_name, meta in meta_map.items():
            ids = meta.get("ObjectIDs", [])
            if object_id not in ids:
                continue
            idx = ids.index(object_id)
            keys = meta.setdefault("AttributeKeys", [])
            vals = meta.setdefault("AttributeValues", [])
            while len(keys) <= idx:
                keys.append([])
            while len(vals) <= idx:
                vals.append([])
            if not isinstance(keys[idx], list):
                keys[idx] = list(keys[idx]) if keys[idx] else []
            if not isinstance(vals[idx], list):
                vals[idx] = list(vals[idx]) if vals[idx] else []
            speed_kmh = speed_ms * 3.6
            speed_str = f"{speed_kmh:.1f}"
            if "Speed" in keys[idx]:
                s_idx = keys[idx].index("Speed")
                if len(vals[idx]) > s_idx:
                    vals[idx][s_idx] = speed_str
                else:
                    vals[idx].append(speed_str)
            else:
                keys[idx].append("Speed")
                vals[idx].append(speed_str)
            return True
        return False
