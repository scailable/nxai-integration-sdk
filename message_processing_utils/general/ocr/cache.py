"""
Cache for OCR recognition results.
"""

import threading
import time
from message_processing_utils.base.messages import InferenceMessage


class OcrCache:
    """Best-confidence OCR result per object.

    Written from OCR worker threads and read/cleaned from the main message
    loop, so every access to the two dicts is guarded by `_lock`.
    """

    def __init__(self, engine, output_name, ocr_pool=None, ttl_sec: float = 300.0):
        self._ocr_results_cache = {}
        self._ocr_timestamps = {}
        self._lock = threading.Lock()
        self.engine = engine
        self.output_name = output_name
        self.ocr_pool = ocr_pool
        self.ttl_sec = ttl_sec

    def cache_ocr_result(self, message: InferenceMessage, recognized_text: str, confidence: float):
        if not message.original_object_id:
            return
        object_id = message.original_object_id
        with self._lock:
            existing = self._ocr_results_cache.get(object_id)
            if existing is None or confidence > existing[1]:
                self._ocr_results_cache[object_id] = (recognized_text, confidence)
            self._ocr_timestamps[object_id] = time.monotonic()

    def get_cached_result(self, object_id: str):
        with self._lock:
            return self._ocr_results_cache.get(object_id)

    def cleanup(self, current_time: float):
        """Remove entries older than ttl_sec."""
        with self._lock:
            expired = [
                oid for oid, ts in self._ocr_timestamps.items()
                if current_time - ts > self.ttl_sec
            ]
            for oid in expired:
                self._ocr_results_cache.pop(oid, None)
                self._ocr_timestamps.pop(oid, None)
