"""
ANPR detector message classes (license plate, speed).
"""

import logging
from message_processing_utils.anpr.attributes import LICENSE_PLATE_CONFIDENCE, LICENSE_PLATE_TEXT
from message_processing_utils.general.detector.messages import DetectorMessage


logger = logging.getLogger(__name__)


class AnprDetectorMessage(DetectorMessage):
    """ANPR-specific detector message with custom attribute names."""

    def __init__(self, sdk_message, raw_tensors=None):
        super().__init__(sdk_message, raw_tensors)

    def handle(self, ocr_cache) -> None:
        for object_id in set(self.object_ids):
            cached = ocr_cache.get_cached_result(object_id)
            if cached:
                text, conf = cached
                self.add_license_plate_metadata(object_id, text, conf)
        logger.debug("Returning detector message with cached OCR results")
        return None

    def add_license_plate_metadata(self, object_id: str, recognized_text: str, confidence: float) -> bool:
        obj = self.find_object(object_id)
        if obj is None:
            return False
        obj.attributes[LICENSE_PLATE_TEXT] = recognized_text
        obj.attributes[LICENSE_PLATE_CONFIDENCE] = f"{float(confidence):.4f}"
        return True


class SpeedDetectorMessage(AnprDetectorMessage):
    """Speed measurement detector message that extends ANPR functionality with speed metadata."""

    def __init__(self, sdk_message, raw_tensors=None):
        super().__init__(sdk_message, raw_tensors)

    def add_speed_metadata(self, object_id: str, speed_ms: float) -> bool:
        """
        Add speed attribute to object metadata.

        Args:
            object_id: Object ID to add speed attribute to.
            speed_ms: Speed in m/s.

        Returns:
            True if speed was added, False otherwise.
        """
        obj = self.find_object(object_id)
        if obj is None:
            return False
        speed_kmh = speed_ms * 3.6
        obj.attributes["Speed"] = f"{speed_kmh:.1f}"
        return True
