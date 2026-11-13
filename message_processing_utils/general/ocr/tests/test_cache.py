import os
import sys
import threading
import time
import unittest

_nxai_utilities = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "nxai-utilities")
)
if _nxai_utilities not in sys.path:
    sys.path.insert(0, _nxai_utilities)

from message_processing_utils.general.ocr.cache import OcrCache


class _Msg:
    def __init__(self, object_id):
        self.original_object_id = object_id


class TestOcrCacheMaxConfidence(unittest.TestCase):

    def _cache(self, ttl_sec=300.0):
        return OcrCache(engine=None, output_name="Identity:0", ttl_sec=ttl_sec)

    def test_first_result_stored(self):
        cache = self._cache()
        cache.cache_ocr_result(_Msg("obj-1"), "ABC", 0.75)
        self.assertEqual(cache.get_cached_result("obj-1"), ("ABC", 0.75))

    def test_higher_confidence_overwrites(self):
        cache = self._cache()
        cache.cache_ocr_result(_Msg("obj-1"), "ABC", 0.75)
        cache.cache_ocr_result(_Msg("obj-1"), "A8C", 0.95)
        self.assertEqual(cache.get_cached_result("obj-1"), ("A8C", 0.95))

    def test_lower_confidence_does_not_overwrite(self):
        cache = self._cache()
        cache.cache_ocr_result(_Msg("obj-1"), "A8C", 0.95)
        cache.cache_ocr_result(_Msg("obj-1"), "ABC", 0.80)
        self.assertEqual(cache.get_cached_result("obj-1"), ("A8C", 0.95))

    def test_equal_confidence_does_not_overwrite(self):
        cache = self._cache()
        cache.cache_ocr_result(_Msg("obj-1"), "A8C", 0.95)
        cache.cache_ocr_result(_Msg("obj-1"), "ABC", 0.95)
        self.assertEqual(cache.get_cached_result("obj-1"), ("A8C", 0.95))

    def test_sequence_from_spec(self):
        # Frame 1: ABC@75, Frame 2: A8C@95, Frame 3: ABC@80 → should return A8C@95
        cache = self._cache()
        cache.cache_ocr_result(_Msg("obj-1"), "ABC", 0.75)
        cache.cache_ocr_result(_Msg("obj-1"), "A8C", 0.95)
        cache.cache_ocr_result(_Msg("obj-1"), "ABC", 0.80)
        self.assertEqual(cache.get_cached_result("obj-1"), ("A8C", 0.95))

    def test_timestamp_updated_on_lower_confidence(self):
        cache = self._cache(ttl_sec=1.0)
        cache.cache_ocr_result(_Msg("obj-1"), "A8C", 0.95)
        # Advance time past TTL by manipulating the stored timestamp
        cache._ocr_timestamps["obj-1"] -= 1.5
        # Lower-confidence write should refresh the timestamp
        cache.cache_ocr_result(_Msg("obj-1"), "ABC", 0.50)
        cache.cleanup(time.monotonic())
        # Entry must still be present (timestamp was refreshed)
        self.assertIsNotNone(cache.get_cached_result("obj-1"))
        # And the result must still be the best one
        self.assertEqual(cache.get_cached_result("obj-1"), ("A8C", 0.95))

    def test_cleanup_removes_expired(self):
        cache = self._cache(ttl_sec=1.0)
        cache.cache_ocr_result(_Msg("obj-1"), "ABC", 0.75)
        cache._ocr_timestamps["obj-1"] -= 2.0
        cache.cleanup(time.monotonic())
        self.assertIsNone(cache.get_cached_result("obj-1"))

    def test_no_object_id_ignored(self):
        cache = self._cache()
        cache.cache_ocr_result(_Msg(None), "ABC", 0.75)
        cache.cache_ocr_result(_Msg(""), "ABC", 0.75)
        self.assertEqual(cache._ocr_results_cache, {})


class TestOcrCacheThreadSafety(unittest.TestCase):
    """OCR results are written from worker threads while the main loop cleans up."""

    def test_cleanup_survives_concurrent_writes(self):
        cache = OcrCache(engine=None, output_name="Identity:0", ttl_sec=30.0)
        errors = []
        stop = threading.Event()
        started = threading.Barrier(5)

        def writer(start):
            started.wait()
            i = start
            while not stop.is_set():
                cache.cache_ocr_result(_Msg(f"obj-{i}"), "ABC", 0.75)
                i += 1

        writers = [threading.Thread(target=writer, args=(n * 1_000_000,)) for n in range(4)]
        for thread in writers:
            thread.start()
        started.wait()
        # cleanup() must iterate a dict large enough that a concurrent insert
        # lands mid-iteration; over a near-empty dict the race cannot be hit.
        while len(cache._ocr_timestamps) < 50_000:
            time.sleep(0.01)
        try:
            for _ in range(200):
                cache.cleanup(time.monotonic())
        except Exception as exc:
            errors.append(exc)
        finally:
            stop.set()
            for thread in writers:
                thread.join()

        self.assertEqual(errors, [])

    def test_concurrent_writes_are_not_lost(self):
        cache = OcrCache(engine=None, output_name="Identity:0")
        barrier = threading.Barrier(4)

        def writer(confidence):
            barrier.wait()
            for _ in range(500):
                cache.cache_ocr_result(_Msg("obj-1"), "ABC", confidence)

        confidences = [0.10, 0.50, 0.80, 0.99]
        threads = [threading.Thread(target=writer, args=(c,)) for c in confidences]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(cache.get_cached_result("obj-1"), ("ABC", 0.99))


if __name__ == "__main__":
    unittest.main()
