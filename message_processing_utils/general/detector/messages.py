"""
General detection message class (bounding boxes, metadata).
"""

import logging
from pprint import pformat
import msgpack
from message_processing_utils.base.messages import InferenceMessage


logger = logging.getLogger(__name__)


class DetectorMessage(InferenceMessage):
    """Message from detector model (with bounding boxes)."""

    def __init__(self, message: dict):
        super().__init__(message)

    def to_bytes(self):
        msg = self._message
        bboxes = self.bboxes_xyxy
        meta = self.objects_metadata
        for class_name, coords in bboxes.items():
            entry = meta.get(class_name, {})
            ids = entry.get("ObjectIDs", [])
            keys = entry.get("AttributeKeys", [])
            vals = entry.get("AttributeValues", [])
            num_boxes = len(coords) // 4
            if len(ids) != num_boxes:
                ids = list(ids[:num_boxes]) + [None] * (num_boxes - len(ids))
            while len(keys) < num_boxes:
                keys.append([])
            while len(vals) < num_boxes:
                vals.append([])
            meta[class_name] = {
                "ObjectIDs": ids,
                "AttributeKeys": keys,
                "AttributeValues": vals,
            }
        logger.debug("Sending message: %s", pformat(msg))
        return msgpack.packb(msg, use_bin_type=True)

    def handle(self, ocr_cache) -> None:
        for object_id in set(self.object_ids):
            cached = ocr_cache.get_cached_result(object_id)
            if cached:
                text, conf = cached
                self.update_metadata(object_id, text, conf)
        logger.debug("Returning detector message with cached OCR results")
        return None

    def update_metadata(self, object_id: str, recognized_text: str, confidence: float) -> bool:
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

            # Update recognized_text
            if "recognized_text" in keys[idx]:
                k_idx = keys[idx].index("recognized_text")
                if len(vals[idx]) > k_idx:
                    vals[idx][k_idx] = recognized_text
                else:
                    vals[idx].append(recognized_text)
            else:
                keys[idx].append("recognized_text")
                vals[idx].append(recognized_text)

            # Update confidence
            if "confidence" in keys[idx]:
                c_idx = keys[idx].index("confidence")
                if len(vals[idx]) > c_idx:
                    vals[idx][c_idx] = str(confidence)
                else:
                    vals[idx].append(str(confidence))
            else:
                keys[idx].append("confidence")
                vals[idx].append(str(confidence))

            return True
        return False
