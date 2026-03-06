"""
Unit tests for SpeedMeasurementCache.
"""

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import sys
import os

# Add repo root so message_processing_utils and speed_cache are importable
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.join(script_dir, "..")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from speed_cache import SpeedMeasurementCache
from message_processing_utils import InferenceMessage


class TestSpeedMeasurementCache(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.camera_1_id = "3645c7ee-ca91-e579-e753-1d85af1fd08c"
        self.camera_2_id = "e3e9a385-7fe0-3ba5-5482-a86cde7faf48"
        self.distance_m = 100.0
        self.timeout_sec = 300.0
        self.cache = SpeedMeasurementCache(
            timeout_sec=self.timeout_sec,
            distance_m=self.distance_m,
            camera_1_id=self.camera_1_id,
            camera_2_id=self.camera_2_id
        )

    def tearDown(self):
        """Clean up after tests."""
        self.cache.stop()

    def test_add_detection_first_camera(self):
        """Test adding detection on first camera."""
        timestamp = datetime.now()
        result = self.cache.add_detection(
            "ABC123",
            self.camera_1_id,
            timestamp
        )
        self.assertIsNone(result)  # No match yet
        self.assertIn("ABC123", self.cache._cache)
        # Camera ID is normalized in cache
        normalized_id = self.cache._camera_1_id
        self.assertEqual(self.cache._cache["ABC123"][normalized_id], timestamp)

    def test_add_detection_same_camera_twice(self):
        """Test that second detection on same camera doesn't update timestamp."""
        timestamp1 = datetime.now()
        timestamp2 = timestamp1 + timedelta(seconds=10)
        
        self.cache.add_detection(
            "ABC123",
            self.camera_1_id,
            timestamp1
        )
        result = self.cache.add_detection(
            "ABC123",
            self.camera_1_id,
            timestamp2
        )
        
        self.assertIsNone(result)
        # First timestamp should be kept
        normalized_id = self.cache._camera_1_id
        self.assertEqual(self.cache._cache["ABC123"][normalized_id], timestamp1)

    def test_cross_camera_match(self):
        """Test speed calculation when plate detected on both cameras."""
        timestamp1 = datetime.now()
        timestamp2 = timestamp1 + timedelta(seconds=10)  # 10 seconds later
        
        # First detection on camera 1
        self.cache.add_detection(
            "ABC123",
            self.camera_1_id,
            timestamp1
        )
        
        # Second detection on camera 2
        result = self.cache.add_detection(
            "ABC123",
            self.camera_2_id,
            timestamp2
        )
        
        # Should calculate speed: 100m / 10s = 10 m/s
        self.assertIsNotNone(result)
        self.assertEqual(result, 10.0)
        # Plate should be removed from cache
        self.assertNotIn("ABC123", self.cache._cache)

    def test_cross_camera_match_reverse_order(self):
        """Test speed calculation when plate detected on camera 2 first."""
        timestamp1 = datetime.now()
        timestamp2 = timestamp1 + timedelta(seconds=5)
        
        # First detection on camera 2
        self.cache.add_detection(
            "XYZ789",
            self.camera_2_id,
            timestamp1
        )
        
        # Second detection on camera 1
        result = self.cache.add_detection(
            "XYZ789",
            self.camera_1_id,
            timestamp2
        )
        
        # Should calculate speed: 100m / 5s = 20 m/s
        self.assertIsNotNone(result)
        self.assertEqual(result, 20.0)
        self.assertNotIn("XYZ789", self.cache._cache)

    def test_ignore_unknown_camera(self):
        """Test that detections from unknown cameras are ignored."""
        timestamp = datetime.now()
        # Use a valid UUID that's not one of our cameras
        result = self.cache.add_detection(
            "ABC123",
            "00000000-0000-0000-0000-000000000000",
            timestamp
        )
        
        self.assertIsNone(result)
        self.assertNotIn("ABC123", self.cache._cache)

    def test_empty_license_plate(self):
        """Test that empty license plates are ignored."""
        timestamp = datetime.now()
        result = self.cache.add_detection(
            "",
            self.camera_1_id,
            timestamp
        )
        
        self.assertIsNone(result)
        self.assertEqual(len(self.cache._cache), 0)

    def test_cleanup_expired(self):
        """Test that expired entries are cleaned up."""
        old_timestamp = datetime.now() - timedelta(seconds=self.timeout_sec + 1)
        
        self.cache.add_detection(
            "OLD123",
            self.camera_1_id,
            old_timestamp
        )
        
        # Manually trigger cleanup
        self.cache._cleanup_expired()
        
        # Entry should be removed
        self.assertNotIn("OLD123", self.cache._cache)

    def test_normalize_camera_id(self):
        """Test that camera IDs are normalized for comparison."""
        timestamp = datetime.now()
        
        # Add with normalized camera ID (as it comes from InferenceMessage.device_id)
        normalized_camera_id = "3645c7ee-ca91-e579-e753-1d85af1fd08c"
        self.cache.add_detection(
            "ABC123",
            normalized_camera_id,
            timestamp
        )
        
        # Should work because camera_id is already normalized
        self.assertIn("ABC123", self.cache._cache)
    
    def test_invalid_time_delta(self):
        """Test handling of invalid time delta (negative or zero)."""
        timestamp1 = datetime.now()
        timestamp2 = timestamp1 - timedelta(seconds=1)  # Earlier timestamp
        
        self.cache.add_detection(
            "ABC123",
            self.camera_1_id,
            timestamp1
        )
        
        # Try to add with earlier timestamp (shouldn't happen, but handle gracefully)
        result = self.cache.add_detection(
            "ABC123",
            self.camera_2_id,
            timestamp2
        )
        
        # Should handle gracefully and not calculate speed
        self.assertIsNone(result)

    def test_update_config_distance_and_timeout_does_not_clear_cache(self):
        """Changing only distance or timeout must not clear caches."""
        ts = datetime.now()
        self.cache.add_detection("ABC123", self.camera_1_id, ts)
        self.assertIn("ABC123", self.cache._cache)
        self.cache.update_config(
            self.camera_1_id,
            self.camera_2_id,
            distance_m=200.0,
            timeout_sec=120.0,
        )
        self.assertIn("ABC123", self.cache._cache)
        self.assertEqual(self.cache._distance_m, 200.0)
        self.assertEqual(self.cache._timeout_sec, 120.0)

    def test_update_config_camera_ids_clears_cache(self):
        """Changing camera IDs must clear both caches."""
        ts = datetime.now()
        self.cache.add_detection("ABC123", self.camera_1_id, ts)
        self.cache.set_speed("XYZ789", 15.0)
        self.assertIn("ABC123", self.cache._cache)
        self.assertIn("XYZ789", self.cache._speed_cache)
        new_cam1 = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
        new_cam2 = "ffffffff-1111-4111-8111-111111111111"
        self.cache.update_config(new_cam1, new_cam2, self.distance_m, self.timeout_sec)
        self.assertNotIn("ABC123", self.cache._cache)
        self.assertNotIn("XYZ789", self.cache._speed_cache)
        self.assertEqual(self.cache._camera_1_id, new_cam1.lower())
        self.assertEqual(self.cache._camera_2_id, new_cam2.lower())


if __name__ == '__main__':
    unittest.main()
