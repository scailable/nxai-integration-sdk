"""
Cache for tracking license plate detections across two cameras for speed measurement.
"""

import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Optional
import sys
import os

# Add message_processing_utils package to path
script_location = os.path.dirname(os.path.abspath(__file__))
parent = os.path.join(script_location, "..")
if parent not in sys.path:
    sys.path.insert(0, parent)

from message_processing_utils import InferenceMessage


logger = logging.getLogger(__name__)


class SpeedMeasurementCache:
    """
    Cache for tracking license plate detections on two cameras.
    
    Tracks when a license plate was first detected on each camera and calculates
    average speed when the same plate is detected on both cameras.
    """

    def __init__(
        self,
        timeout_sec: float,
        distance_m: float,
        camera_1_id: str,
        camera_2_id: str,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        Initialize speed measurement cache.
        
        Args:
            timeout_sec: Timeout in seconds for cache entries to expire.
            distance_m: Distance between cameras in meters.
            camera_1_id: Device ID (UUID or string) of the first camera.
            camera_2_id: Device ID (UUID or string) of the second camera.
            logger_instance: Logger instance. If None, uses module logger.
        """
        # {license_plate: {camera_id: first_seen_time}}
        self._cache: Dict[str, Dict[str, datetime]] = {}
        # {license_plate: (speed_ms, last_seen_time)}
        self._speed_cache: Dict[str, tuple[float, datetime]] = {}
        self._timeout_sec = timeout_sec
        self._distance_m = distance_m
        self._camera_1_id = InferenceMessage.normalize_device_id(camera_1_id)
        self._camera_2_id = InferenceMessage.normalize_device_id(camera_2_id)
        self._cleanup_interval_sec = 15.0
        self._last_cleanup = datetime.now()
        self._lock = threading.RLock()
        self._logger = logger_instance or logger
        self._cleanup_timer: Optional[threading.Timer] = None
        self._stop_event = threading.Event()
        self._start_cleanup_timer()

    def update_config(
        self,
        camera_1_id: str,
        camera_2_id: str,
        distance_m: float,
        timeout_sec: float,
    ) -> None:
        """
        Update camera IDs, distance and timeout. Clears caches if camera set changed.
        Thread-safe.
        """
        new_cam1 = InferenceMessage.normalize_device_id(camera_1_id)
        new_cam2 = InferenceMessage.normalize_device_id(camera_2_id)
        with self._lock:
            cameras_changed = (
                new_cam1 != self._camera_1_id or new_cam2 != self._camera_2_id
            )
            self._camera_1_id = new_cam1
            self._camera_2_id = new_cam2
            self._distance_m = distance_m
            self._timeout_sec = timeout_sec
            if cameras_changed:
                self._cache.clear()
                self._speed_cache.clear()
                self._logger.debug(
                    "Cleared speed caches after camera config change"
                )

    def _start_cleanup_timer(self) -> None:
        """Start periodic cleanup timer."""
        if self._stop_event.is_set():
            return
        self._cleanup_timer = threading.Timer(
            self._cleanup_interval_sec, self._cleanup_expired
        )
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()
        self._logger.debug(
            "Started cleanup timer (interval: %.1f seconds)",
            self._cleanup_interval_sec
        )

    def add_detection(
        self,
        license_plate: str,
        camera_id: str,
        timestamp: datetime
    ) -> Optional[float]:
        """
        Add a license plate detection and check for cross-camera match.
        
        Args:
            license_plate: Recognized license plate text.
            camera_id: Camera ID where detection occurred (will be normalized).
            timestamp: Timestamp of detection.
        
        Returns:
            Average speed in m/s if match found, None otherwise.
        """
        if not license_plate or not license_plate.strip():
            return None
        if camera_id not in (self._camera_1_id, self._camera_2_id):
            self._logger.debug(
                "Ignoring detection from unknown camera: %s",
                camera_id
            )
            return None
        with self._lock:
            if license_plate not in self._cache:
                self._cache[license_plate] = {camera_id: timestamp}
                self._logger.debug(
                    "Added new license plate detection: %s on camera %s",
                    license_plate, camera_id
                )
                return None
            camera_times = self._cache[license_plate]
            if camera_id in camera_times:
                self._logger.debug(
                    "License plate %s already seen on camera %s, "
                    "keeping first timestamp",
                    license_plate, camera_id
                )
                return None
            other_camera_id = (
                self._camera_1_id
                if camera_id == self._camera_2_id
                else self._camera_2_id
            )
            if other_camera_id not in camera_times:
                self._cache[license_plate][camera_id] = timestamp
                self._logger.debug(
                    "Added detection for license plate %s on camera %s "
                    "(waiting for other camera)",
                    license_plate, camera_id
                )
                return None
            other_camera_time = camera_times[other_camera_id]
            time_delta = timestamp - other_camera_time
            if time_delta.total_seconds() <= 0:
                self._logger.warning(
                    "Invalid time delta for plate %s: %s. Removing old entry and adding new one.",
                    license_plate, time_delta
                )
                del self._cache[license_plate]
                self._cache[license_plate] = {camera_id: timestamp}
                return None
            avg_speed = self._distance_m / time_delta.total_seconds()
            self._logger.info(
                "Average speed of the car with license plate %s was %d m/s",
                license_plate, int(avg_speed)
            )
            self.set_speed(license_plate, avg_speed)
            del self._cache[license_plate]
            self._logger.debug(
                "Removed license plate %s from cache after speed calculation",
                license_plate
            )
            return avg_speed

    def set_speed(self, license_plate: str, speed_ms: float) -> None:
        """
        Store speed for a license plate.
        
        Args:
            license_plate: License plate text.
            speed_ms: Speed in m/s.
        """
        if not license_plate or not license_plate.strip():
            return
        with self._lock:
            self._speed_cache[license_plate.strip()] = (speed_ms, datetime.now())
            self._logger.debug(
                "Stored speed %.1f m/s for license plate %s",
                speed_ms, license_plate
            )

    def get_speed(self, license_plate: str) -> Optional[float]:
        """
        Get speed for a license plate.
        
        Args:
            license_plate: License plate text.
        
        Returns:
            Speed in m/s if found, None otherwise.
        """
        if not license_plate or not license_plate.strip():
            return None
        with self._lock:
            entry = self._speed_cache.get(license_plate.strip())
            if entry is None:
                return None
            speed_ms, last_seen = entry
            if (datetime.now() - last_seen).total_seconds() >= self._timeout_sec:
                del self._speed_cache[license_plate.strip()]
                return None
            return speed_ms

    def update_last_seen(self, license_plate: str) -> None:
        """
        Update last seen time for a license plate.
        
        This is called when an object with this license plate is detected
        in a new message, extending the cache lifetime.
        
        Args:
            license_plate: License plate text.
        """
        if not license_plate or not license_plate.strip():
            return
        with self._lock:
            entry = self._speed_cache.get(license_plate.strip())
            if entry is not None:
                speed_ms, _ = entry
                self._speed_cache[license_plate.strip()] = (speed_ms, datetime.now())

    def _cleanup_expired(self):
        """Remove expired cache entries."""
        if self._stop_event.is_set():
            return
        now = datetime.now()
        with self._lock:
            expired_count = 0
            plates_to_remove = []
            for license_plate, camera_times in list(self._cache.items()):
                expired_cameras = []
                for camera_id, camera_time in camera_times.items():
                    if (now - camera_time).total_seconds() >= self._timeout_sec:
                        expired_cameras.append(camera_id)
                
                for camera_id in expired_cameras:
                    del camera_times[camera_id]
                    expired_count += 1
                    self._logger.debug(
                        "Removed expired detection for plate %s on camera %s",
                        license_plate, camera_id
                    )
                if not camera_times:
                    plates_to_remove.append(license_plate)
            for plate in plates_to_remove:
                del self._cache[plate]
            speed_expired_count = 0
            speed_plates_to_remove = []
            for license_plate, (speed_ms, last_seen) in list(self._speed_cache.items()):
                if (now - last_seen).total_seconds() >= self._timeout_sec:
                    speed_plates_to_remove.append(license_plate)
                    speed_expired_count += 1
            for plate in speed_plates_to_remove:
                del self._speed_cache[plate]
            if expired_count > 0 or speed_expired_count > 0:
                self._logger.debug(
                    "Cleaned up %d expired detections, removed %d empty plates, "
                    "cleaned up %d expired speed entries",
                    expired_count, len(plates_to_remove), speed_expired_count
                )
        self._start_cleanup_timer()

    def stop(self):
        """Stop the cleanup timer."""
        self._stop_event.set()
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
            self._cleanup_timer = None
