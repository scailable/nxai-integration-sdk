"""
General detection message class (bounding boxes, metadata).
"""

import logging
from message_processing_utils.base.messages import InferenceMessage


logger = logging.getLogger(__name__)


class DetectorMessage(InferenceMessage):
    """Message from detector model (with bounding boxes)."""

    def __init__(self, sdk_message, raw_tensors=None):
        super().__init__(sdk_message, raw_tensors)

    def handle(self, ocr_cache) -> None:
        for object_id in set(self.object_ids):
            cached = ocr_cache.get_cached_result(object_id)
            if cached:
                text, conf = cached
                self.update_metadata(object_id, text, conf)
        logger.debug("Returning detector message with cached OCR results")
        return None

    def update_metadata(self, object_id: str, recognized_text: str, confidence: float) -> bool:
        obj = self.find_object(object_id)
        if obj is None:
            return False
        obj.attributes["recognized_text"] = recognized_text
        obj.attributes["confidence"] = str(confidence)
        return True
