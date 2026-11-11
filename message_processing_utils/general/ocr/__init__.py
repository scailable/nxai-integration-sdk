"""
General OCR processing (engine, cache, worker pool, config).
"""

from message_processing_utils.general.ocr.messages import OcrMessage
from message_processing_utils.general.ocr.engine import LogitsOcrEngine
from message_processing_utils.general.ocr.worker_pool import OcrWorkerPool
from message_processing_utils.general.ocr.cache import OcrCache

__all__ = [
    "OcrMessage",
    "LogitsOcrEngine",
    "OcrWorkerPool",
    "OcrCache",
]
