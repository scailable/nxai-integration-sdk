"""
Cache for OCR recognition results.
"""

from message_processing_utils.base.messages import InferenceMessage


class OcrCache:

    def __init__(self, engine, output_name, ocr_pool=None):
        self._ocr_results_cache = {}
        self.engine = engine
        self.output_name = output_name
        self.ocr_pool = ocr_pool

    def cache_ocr_result(self, message: InferenceMessage, recognized_text: str, confidence: float):
        if not message.original_object_id:
            return
        self._ocr_results_cache[message.original_object_id] = (recognized_text, confidence)

    def get_cached_result(self, object_id: str):
        return self._ocr_results_cache.get(object_id)
