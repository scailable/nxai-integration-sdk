"""
Worker pool for asynchronous OCR processing.
"""

import logging
import threading
import queue
from typing import Optional


logger = logging.getLogger(__name__)


class OcrWorkerPool:

    def __init__(self, engine, worker_count: int):
        self._engine = engine
        self._queue: queue.Queue[Optional[tuple]] = queue.Queue()
        self._stop_event = threading.Event()
        self._threads = []
        logger.info("OCR worker pool starting with %d workers", worker_count)
        for idx in range(worker_count):
            thread = threading.Thread(target=self._worker_loop, name=f"ocr-worker-{idx}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def submit(self, logits, on_done):
        self._queue.put((logits, on_done))

    def stop(self):
        logger.info("Stopping OCR worker pool")
        self._stop_event.set()
        for _ in self._threads:
            self._queue.put(None)
        for thread in self._threads:
            thread.join()

    def _worker_loop(self):
        logger.debug("OCR worker thread started")
        while not self._stop_event.is_set():
            item = self._queue.get()
            if item is None:
                logger.debug("OCR worker thread exiting")
                return
            logits, on_done = item
            try:
                result = self._engine.decode_logits(logits)
                on_done(result, None)
            except Exception as exc:
                on_done(None, exc)
