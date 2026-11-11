import os
import sys
import unittest

_nxai_utilities = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "nxai-utilities")
)
if _nxai_utilities not in sys.path:
    sys.path.insert(0, _nxai_utilities)

from message_processing_utils.anpr.detector.event_cache import EventDeduplicationCache


class TestEventDeduplicationCache(unittest.TestCase):

    def setUp(self):
        self.cache = EventDeduplicationCache(ttl_sec=60.0)

    def test_first_occurrence_should_generate(self):
        self.assertTrue(self.cache.should_generate_event("obj-1", "ABC123"))

    def test_same_object_same_text_should_not_generate(self):
        self.cache.mark_event_generated("obj-1", "ABC123", current_time=0.0)
        self.assertFalse(self.cache.should_generate_event("obj-1", "ABC123"))

    def test_same_object_different_text_should_generate(self):
        self.cache.mark_event_generated("obj-1", "ABC123", current_time=0.0)
        self.assertTrue(self.cache.should_generate_event("obj-1", "XYZ999"))

    def test_different_object_same_text_should_generate(self):
        self.cache.mark_event_generated("obj-1", "ABC123", current_time=0.0)
        self.assertTrue(self.cache.should_generate_event("obj-2", "ABC123"))

    def test_expired_entry_should_generate_again(self):
        self.cache.mark_event_generated("obj-1", "ABC123", current_time=0.0)
        self.cache.cleanup(current_time=61.0)  # 61s > ttl_sec=60
        self.assertTrue(self.cache.should_generate_event("obj-1", "ABC123"))

    def test_cleanup_keeps_unexpired_entries(self):
        self.cache.mark_event_generated("obj-1", "ABC123", current_time=0.0)
        self.cache.mark_event_generated("obj-2", "XYZ999", current_time=50.0)
        self.cache.cleanup(current_time=61.0)  # obj-1 expired, obj-2 not
        self.assertTrue(self.cache.should_generate_event("obj-1", "ABC123"))
        self.assertFalse(self.cache.should_generate_event("obj-2", "XYZ999"))

    def test_mark_updates_text_and_timestamp(self):
        self.cache.mark_event_generated("obj-1", "ABC123", current_time=0.0)
        self.cache.mark_event_generated("obj-1", "XYZ999", current_time=1.0)
        self.assertFalse(self.cache.should_generate_event("obj-1", "XYZ999"))
        self.assertTrue(self.cache.should_generate_event("obj-1", "ABC123"))

    def test_cleanup_empty_cache_is_noop(self):
        self.cache.cleanup(current_time=1000.0)  # should not raise


if __name__ == "__main__":
    unittest.main()
