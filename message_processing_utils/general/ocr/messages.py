"""
Base OCR message class.
"""

import logging
from abc import ABC, abstractmethod
from message_processing_utils.base.messages import InferenceMessage


logger = logging.getLogger(__name__)


class OcrMessage(InferenceMessage, ABC):
    """Message from OCR model (with BinaryOutputs)."""

    def __init__(self, sdk_message, raw_tensors=None):
        super().__init__(sdk_message, raw_tensors)

    @abstractmethod
    def extract_logits_array(self, output_name):
        """Return the named tensor as a logits array, or None if unusable."""

    @abstractmethod
    def decode_ocr_logits(self, engine, output_name):
        """Decode OCR logits and return recognized text."""

    def handle(self, ocr_cache) -> None:
        if ocr_cache.ocr_pool:
            ocr_logits = self.extract_logits_array(ocr_cache.output_name)
            if ocr_logits is None:
                return
            def on_done(result, error):
                if error is not None:
                    logger.warning("OCR worker error: %s", error)
                elif result:
                    text, conf = result
                    ocr_cache.cache_ocr_result(self, text, conf)
            ocr_cache.ocr_pool.submit(ocr_logits, on_done)
        else:
            result = self.decode_ocr_logits(ocr_cache.engine, ocr_cache.output_name)
            if result:
                text, conf = result
                ocr_cache.cache_ocr_result(self, text, conf)
        return None
