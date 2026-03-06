"""
Unit tests for postprocessor-python-measure-average-car-speed.
"""

import unittest
import logging
from unittest.mock import MagicMock, patch, mock_open
import sys
import os
import numpy as np
from datetime import datetime

# Add the nxai-utilities path before importing the module
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../nxai-utilities/python-utilities"))
# Add repo root so message_processing_utils is importable
repo_root = os.path.join(script_dir, "..")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Add current directory to path for speed_cache import
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Define mock exception classes that inherit from Exception
class MockSocketTimeout(Exception):
    pass

class MockSocketError(Exception):
    pass

# Mock dependencies before importing the main module
mock_comm = MagicMock()
mock_comm.SocketTimeout = MockSocketTimeout
mock_comm.SocketError = MockSocketError
sys.modules['nxai_communication_utils'] = mock_comm
sys.modules['msgpack'] = MagicMock()

import logging as _logging

# Patch the FileHandler to avoid creating log files during tests
with patch('logging.FileHandler') as _file_handler:
    _file_handler.return_value = _logging.NullHandler()
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "postprocessor_python_measure_average_car_speed",
        os.path.join(script_dir, "postprocessor-python-measure-average-car-speed.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["postprocessor_python_measure_average_car_speed"] = module
    # Ensure nxai_communication_utils is available in the module's namespace
    module.nxai_communication_utils = mock_comm
    spec.loader.exec_module(module)
    
    # Import classes from message_processing_utils
    from message_processing_utils import InferenceMessage, GenericMessage
    from message_processing_utils.general.detector import DetectorMessage
    from message_processing_utils.anpr import SpeedDetectorMessage, CctOcrMessage
    from message_processing_utils.general.ocr import (
        OcrCache,
        LogitsOcrEngine,
        OcrWorkerPool,
        load_ocr_config,
    )
    from speed_cache import SpeedMeasurementCache


class TestExtractLicensePlates(unittest.TestCase):
    """Tests for _extract_license_plates_from_message() function."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detection_time = datetime.now()
    
    def test_extract_license_plates_with_timestamp(self):
        """Test extracting license plates from DetectorMessage with valid timestamp."""
        message = DetectorMessage({
            "DeviceID": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
            "Timestamp": self.detection_time.timestamp(),
            "BBoxes_xyxy": {"lp": [0, 0, 10, 10]},
            "ObjectsMetaData": {
                "lp": {
                    "ObjectIDs": ["obj-1"],
                    "AttributeKeys": [["License Plate Text", "Confidence"]],
                    "AttributeValues": [["ABC123", "0.95"]]
                }
            }
        })
        results = module._extract_license_plates_from_message(message)
        self.assertEqual(len(results), 1)
        object_id, license_plate, device_id, timestamp = results[0]
        self.assertEqual(object_id, "obj-1")
        self.assertEqual(license_plate, "ABC123")
        self.assertEqual(device_id, "3645c7ee-ca91-e579-e753-1d85af1fd08c")
        self.assertIsInstance(timestamp, datetime)
    
    def test_extract_license_plates_no_timestamp(self):
        """Test handling message without timestamp."""
        message = DetectorMessage({
            "DeviceID": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
            "BBoxes_xyxy": {"lp": [0, 0, 10, 10]},
            "ObjectsMetaData": {
                "lp": {
                    "ObjectIDs": ["obj-1"],
                    "AttributeKeys": [["recognized_text"]],
                    "AttributeValues": [["ABC123"]]
                }
            }
        })
        
        results = module._extract_license_plates_from_message(message)
        self.assertEqual(results, [])
    
    def test_extract_license_plates_invalid_timestamp(self):
        """Test handling invalid timestamp type."""
        message = DetectorMessage({
            "DeviceID": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
            "Timestamp": datetime.now().timestamp(),  # Valid timestamp in message
            "BBoxes_xyxy": {"lp": [0, 0, 10, 10]},
            "ObjectsMetaData": {
                "lp": {
                    "ObjectIDs": ["obj-1"],
                    "AttributeKeys": [["recognized_text"]],
                    "AttributeValues": [["ABC123"]]
                }
            }
        })
        
        # Patch timestamp property to return non-datetime value
        with patch.object(type(message), 'timestamp', new_callable=unittest.mock.PropertyMock) as mock_timestamp:
            mock_timestamp.return_value = "invalid-timestamp"
            with patch.object(module.logger, 'warning') as mock_warning:
                results = module._extract_license_plates_from_message(message)
                self.assertEqual(results, [])
                mock_warning.assert_called_once()
    
    def test_extract_license_plates_multiple_plates(self):
        """Test extracting multiple license plates from one message."""
        message = DetectorMessage({
            "DeviceID": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
            "Timestamp": self.detection_time.timestamp(),
            "BBoxes_xyxy": {"lp": [0, 0, 10, 10, 20, 20, 30, 30]},
            "ObjectsMetaData": {
                "lp": {
                    "ObjectIDs": ["obj-1", "obj-2"],
                    "AttributeKeys": [
                        ["License Plate Text", "Confidence"],
                        ["License Plate Text", "Confidence"]
                    ],
                    "AttributeValues": [
                        ["ABC123", "0.95"],
                        ["XYZ789", "0.88"]
                    ]
                }
            }
        })
        
        results = module._extract_license_plates_from_message(message)
        self.assertEqual(len(results), 2)
        plates = [r[1] for r in results]
        self.assertIn("ABC123", plates)
        self.assertIn("XYZ789", plates)
    
    def test_extract_license_plates_no_recognized_text(self):
        """Test handling message without recognized_text attribute."""
        message = DetectorMessage({
            "DeviceID": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
            "Timestamp": self.detection_time.timestamp(),
            "BBoxes_xyxy": {"lp": [0, 0, 10, 10]},
            "ObjectsMetaData": {
                "lp": {
                    "ObjectIDs": ["obj-1"],
                    "AttributeKeys": [["confidence"]],
                    "AttributeValues": [["0.95"]]
                }
            }
        })
        
        results = module._extract_license_plates_from_message(message)
        self.assertEqual(results, [])
    
    def test_extract_license_plates_empty_text(self):
        """Test handling empty license plate text."""
        message = DetectorMessage({
            "DeviceID": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
            "Timestamp": self.detection_time.timestamp(),
            "BBoxes_xyxy": {"lp": [0, 0, 10, 10]},
            "ObjectsMetaData": {
                "lp": {
                    "ObjectIDs": ["obj-1"],
                    "AttributeKeys": [["recognized_text"]],
                    "AttributeValues": [[""]]
                }
            }
        })
        
        results = module._extract_license_plates_from_message(message)
        self.assertEqual(results, [])
    
    def test_extract_license_plates_invalid_structure(self):
        """Test handling invalid metadata structure."""
        message = DetectorMessage({
            "DeviceID": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
            "Timestamp": self.detection_time.timestamp(),
            "BBoxes_xyxy": {"lp": [0, 0, 10, 10]},
            "ObjectsMetaData": {
                "lp": {
                    "ObjectIDs": ["obj-1"],
                    "AttributeKeys": "invalid",  # Should be list
                    "AttributeValues": [["ABC123"]]
                }
            }
        })
        
        results = module._extract_license_plates_from_message(message)
        self.assertEqual(results, [])


class TestMergeSpeedSettings(unittest.TestCase):
    """Tests for _merge_speed_settings()."""

    def test_merge_uses_ini_when_no_ui(self):
        """When ui_settings is empty, INI values are returned."""
        settings = {
            "camera_1_id": "cam1",
            "camera_2_id": "cam2",
            "distance_between_cameras_m": 50.0,
            "second_appearance_timeout_sec": 30.0,
        }
        result = module._merge_speed_settings(settings, {})
        self.assertEqual(result["camera_1_id"], "cam1")
        self.assertEqual(result["camera_2_id"], "cam2")
        self.assertEqual(result["distance_between_cameras_m"], 50.0)
        self.assertEqual(result["second_appearance_timeout_sec"], 30.0)

    def test_merge_ui_overrides_ini(self):
        """UI settings override INI when present and valid."""
        settings = {
            "camera_1_id": "ini_cam1",
            "camera_2_id": "ini_cam2",
            "distance_between_cameras_m": 100.0,
            "second_appearance_timeout_sec": 60.0,
        }
        ui = {
            module.UI_SETTINGS_PREFIX + "camera_1_id": "ui_cam1",
            module.UI_SETTINGS_PREFIX + "camera_2_id": "ui_cam2",
            module.UI_SETTINGS_PREFIX + "distance_between_cameras_m": 200.0,
            module.UI_SETTINGS_PREFIX + "second_appearance_timeout_sec": 120.0,
        }
        result = module._merge_speed_settings(settings, ui)
        self.assertEqual(result["camera_1_id"], "ui_cam1")
        self.assertEqual(result["camera_2_id"], "ui_cam2")
        self.assertEqual(result["distance_between_cameras_m"], 200.0)
        self.assertEqual(result["second_appearance_timeout_sec"], 120.0)

    def test_merge_invalid_ui_number_keeps_ini(self):
        """Invalid UI number leaves INI value."""
        settings = {
            "camera_1_id": "c1",
            "camera_2_id": "c2",
            "distance_between_cameras_m": 80.0,
            "second_appearance_timeout_sec": 40.0,
        }
        ui = {
            module.UI_SETTINGS_PREFIX + "distance_between_cameras_m": "not_a_number",
        }
        result = module._merge_speed_settings(settings, ui)
        self.assertEqual(result["distance_between_cameras_m"], 80.0)


class TestMainFiltering(unittest.TestCase):
    """Tests for main() function - DeviceID filtering."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.settings = {
            "socket_path": "/tmp/test.sock",
            "camera_1_id": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
            "camera_2_id": "e3e9a385-7fe0-3ba5-5482-a86cde7faf48",
            "ocr_output_name": "Identity:0",
            "nxai_utilities_path": os.path.join(script_dir, "../nxai-utilities/python-utilities")
        }
        self.engine = MagicMock()
    
    @patch('nxai_communication_utils.SocketListener')
    def test_main_filters_by_device_id(self, mock_listener_cls):
        """Test that messages from unknown cameras are returned unchanged."""
        mock_server = mock_listener_cls.return_value
        mock_conn = MagicMock()
        
        # Create message from unknown camera
        unknown_message = GenericMessage({
            "DeviceID": "00000000-0000-0000-0000-000000000000",
            "Timestamp": datetime.now().timestamp()
        })
        
        mock_server.accept.side_effect = [
            (mock_conn, b"raw_data"),
            KeyboardInterrupt
        ]
        
        with patch.object(module, 'create_anpr_message_from_bytes', return_value=unknown_message):
            with self.assertRaises(KeyboardInterrupt):
                module.main(self.settings, self.engine)
            
            # Message should be sent back unchanged
            mock_conn.send.assert_called_once()
            mock_conn.close.assert_called()
    
    @patch('nxai_communication_utils.SocketListener')
    def test_main_processes_known_camera(self, mock_listener_cls):
        """Test that messages from known cameras are processed."""
        mock_server = mock_listener_cls.return_value
        mock_conn = MagicMock()
        
        # Create message from known camera
        known_message = DetectorMessage({
            "DeviceID": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
            "Timestamp": datetime.now().timestamp(),
            "BBoxes_xyxy": {"lp": [0, 0, 10, 10]},
            "ObjectsMetaData": {}
        })
        
        mock_server.accept.side_effect = [
            (mock_conn, b"raw_data"),
            KeyboardInterrupt
        ]
        
        with patch.object(module, 'create_anpr_message_from_bytes', return_value=known_message):
            with patch.object(known_message, 'handle') as mock_handle:
                with patch.object(SpeedDetectorMessage, 'to_bytes', return_value=b"processed"):
                    with self.assertRaises(KeyboardInterrupt):
                        module.main(self.settings, self.engine)
                    mock_handle.assert_called_once()
                    mock_conn.send.assert_called_once_with(b"processed")
    
    @patch('nxai_communication_utils.SocketListener')
    def test_main_handles_socket_timeout(self, mock_listener_cls):
        """Test handling SocketTimeout exceptions."""
        mock_server = mock_listener_cls.return_value
        mock_server.accept.side_effect = [
            MockSocketTimeout,
            KeyboardInterrupt
        ]
        
        with self.assertRaises(KeyboardInterrupt):
            module.main(self.settings, self.engine)
    
    @patch('nxai_communication_utils.SocketListener')
    def test_main_handles_send_error(self, mock_listener_cls):
        """Test handling errors when sending response."""
        mock_server = mock_listener_cls.return_value
        mock_conn = MagicMock()
        mock_conn.send.side_effect = Exception("Send failed")
        
        known_message = DetectorMessage({
            "DeviceID": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
            "Timestamp": datetime.now().timestamp(),
            "BBoxes_xyxy": {"lp": [0, 0, 10, 10]},
            "ObjectsMetaData": {}
        })
        
        mock_server.accept.side_effect = [
            (mock_conn, b"raw_data"),
            KeyboardInterrupt
        ]
        
        with patch.object(module, 'create_anpr_message_from_bytes', return_value=known_message):
            with patch.object(known_message, 'handle'):
                with patch.object(known_message, 'to_bytes', return_value=b"processed"):
                    with patch.object(module.logger, 'warning') as mock_warning:
                        with self.assertRaises(KeyboardInterrupt):
                            module.main(self.settings, self.engine)
                        
                        mock_warning.assert_called()
                        mock_conn.close.assert_called()


class TestMainSpeedCache(unittest.TestCase):
    """Tests for main() function - SpeedMeasurementCache integration."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.settings = {
            "socket_path": "/tmp/test.sock",
            "camera_1_id": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
            "camera_2_id": "e3e9a385-7fe0-3ba5-5482-a86cde7faf48",
            "ocr_output_name": "Identity:0",
            "nxai_utilities_path": os.path.join(script_dir, "../nxai-utilities/python-utilities")
        }
        self.engine = MagicMock()
        self.speed_cache = MagicMock(spec=SpeedMeasurementCache)
    
    @patch('nxai_communication_utils.SocketListener')
    def test_main_adds_detections_to_speed_cache(self, mock_listener_cls):
        """Test that license plates are added to speed_cache for DetectorMessage."""
        mock_server = mock_listener_cls.return_value
        mock_conn = MagicMock()
        
        detection_time = datetime.now()
        message = DetectorMessage({
            "DeviceID": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
            "Timestamp": detection_time.timestamp(),
            "BBoxes_xyxy": {"lp": [0, 0, 10, 10]},
            "ObjectsMetaData": {
                "lp": {
                    "ObjectIDs": ["obj-1"],
                    "AttributeKeys": [["License Plate Text", "Confidence"]],
                    "AttributeValues": [["ABC123", "0.95"]]
                }
            }
        })
        self.speed_cache.add_detection.return_value = None
        self.speed_cache.get_speed.return_value = None
        mock_server.accept.side_effect = [
            (mock_conn, b"raw_data"),
            KeyboardInterrupt
        ]
        with patch.object(module, 'create_anpr_message_from_bytes', return_value=message):
            with patch.object(message, 'handle'):
                with patch.object(message, 'to_bytes', return_value=b"processed"):
                    with self.assertRaises(KeyboardInterrupt):
                        module.main(self.settings, self.engine, speed_cache=self.speed_cache)
                    self.speed_cache.add_detection.assert_called_once_with(
                        "ABC123",
                        "3645c7ee-ca91-e579-e753-1d85af1fd08c",
                        unittest.mock.ANY  # datetime object
                    )
    
    @patch('nxai_communication_utils.SocketListener')
    def test_main_skips_speed_cache_for_non_detector(self, mock_listener_cls):
        """Test that speed_cache is not used for non-DetectorMessage."""
        mock_server = mock_listener_cls.return_value
        mock_conn = MagicMock()
        
        ocr_message = CctOcrMessage({
            "DeviceID": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
            "Timestamp": datetime.now().timestamp(),
            "OriginalObjectID": "obj-1",
            "BinaryOutputs": []
        })
        
        mock_server.accept.side_effect = [
            (mock_conn, b"raw_data"),
            KeyboardInterrupt
        ]
        
        with patch.object(module, 'create_anpr_message_from_bytes', return_value=ocr_message):
            with patch.object(ocr_message, 'handle'):
                with patch.object(ocr_message, 'to_bytes', return_value=b"processed"):
                    with self.assertRaises(KeyboardInterrupt):
                        module.main(self.settings, self.engine, speed_cache=self.speed_cache)
                    
                    # Speed cache should not be called
                    self.speed_cache.add_detection.assert_not_called()

    @patch('nxai_communication_utils.SocketListener')
    def test_main_calls_update_config_when_external_processor_settings_present(
        self, mock_listener_cls
    ):
        """When message has ExternalProcessorSettings, speed_cache.update_config is called."""
        mock_server = mock_listener_cls.return_value
        mock_conn = MagicMock()
        detection_time = datetime.now()
        message = DetectorMessage({
            "DeviceID": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
            "Timestamp": detection_time.timestamp(),
            "BBoxes_xyxy": {"lp": [0, 0, 10, 10]},
            "ObjectsMetaData": {
                "lp": {
                    "ObjectIDs": ["obj-1"],
                    "AttributeKeys": [["License Plate Text", "confidence"]],
                    "AttributeValues": [["ABC123", "0.95"]],
                }
            },
            "ExternalProcessorSettings": {
                module.UI_SETTINGS_PREFIX + "camera_1_id": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
                module.UI_SETTINGS_PREFIX + "camera_2_id": "e3e9a385-7fe0-3ba5-5482-a86cde7faf48",
                module.UI_SETTINGS_PREFIX + "distance_between_cameras_m": 150.0,
                module.UI_SETTINGS_PREFIX + "second_appearance_timeout_sec": 90.0,
            },
        })
        self.speed_cache.add_detection.return_value = None
        self.speed_cache.get_speed.return_value = None
        mock_server.accept.side_effect = [
            (mock_conn, b"raw_data"),
            KeyboardInterrupt
        ]
        with patch.object(module, 'create_anpr_message_from_bytes', return_value=message):
            with patch.object(message, 'handle'):
                with patch.object(message, 'to_bytes', return_value=b"processed"):
                    with self.assertRaises(KeyboardInterrupt):
                        module.main(self.settings, self.engine, speed_cache=self.speed_cache)
        self.speed_cache.update_config.assert_called_once()
        call_args = self.speed_cache.update_config.call_args[0]
        self.assertEqual(call_args[0], "3645c7ee-ca91-e579-e753-1d85af1fd08c")
        self.assertEqual(call_args[1], "e3e9a385-7fe0-3ba5-5482-a86cde7faf48")
        self.assertEqual(call_args[2], 150.0)
        self.assertEqual(call_args[3], 90.0)
    
    @patch('nxai_communication_utils.SocketListener')
    def test_main_skips_speed_cache_when_none(self, mock_listener_cls):
        """Test that processing continues without errors when speed_cache is None."""
        mock_server = mock_listener_cls.return_value
        mock_conn = MagicMock()
        
        message = DetectorMessage({
            "DeviceID": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
            "Timestamp": datetime.now().timestamp(),
            "BBoxes_xyxy": {"lp": [0, 0, 10, 10]},
            "ObjectsMetaData": {
                "lp": {
                    "ObjectIDs": ["obj-1"],
                    "AttributeKeys": [["recognized_text"]],
                    "AttributeValues": [["ABC123"]]
                }
            }
        })
        
        mock_server.accept.side_effect = [
            (mock_conn, b"raw_data"),
            KeyboardInterrupt
        ]
        
        with patch.object(module, 'create_anpr_message_from_bytes', return_value=message):
            with patch.object(message, 'handle'):
                with patch.object(message, 'to_bytes', return_value=b"processed"):
                    # Should not raise any errors
                    with self.assertRaises(KeyboardInterrupt):
                        module.main(self.settings, self.engine, speed_cache=None)
    
    @patch('nxai_communication_utils.SocketListener')
    def test_main_generates_event_when_speed_calculated(self, mock_listener_cls):
        """Test that event is added to message when speed is calculated."""
        mock_server = mock_listener_cls.return_value
        mock_conn = MagicMock()
        
        detection_time = datetime.now()
        message = DetectorMessage({
            "DeviceID": "e3e9a385-7fe0-3ba5-5482-a86cde7faf48",
            "Timestamp": detection_time.timestamp(),
            "BBoxes_xyxy": {"lp": [0, 0, 10, 10]},
            "ObjectsMetaData": {
                "lp": {
                    "ObjectIDs": ["obj-1"],
                    "AttributeKeys": [["License Plate Text", "Confidence"]],
                    "AttributeValues": [["ABC123", "0.95"]]
                }
            }
        })
        self.speed_cache.add_detection.return_value = 20.0
        self.speed_cache.get_speed.return_value = None
        mock_server.accept.side_effect = [
            (mock_conn, b"raw_data"),
            KeyboardInterrupt
        ]
        with patch.object(module, 'create_anpr_message_from_bytes', return_value=message):
            with patch.object(message, 'handle'):
                with patch.object(message, 'to_bytes', return_value=b"processed"):
                    with self.assertRaises(KeyboardInterrupt):
                        module.main(self.settings, self.engine, speed_cache=self.speed_cache)
                    self.assertIn("Events", message._message)
                    events = message._message["Events"]
                    self.assertEqual(len(events), 1)
                    event = events[0]
                    self.assertEqual(event["ID"], "speed.measurement")
                    self.assertEqual(event["Caption"], "Speed Measurement")
                    self.assertIn("ABC123", event["Description"])
                    self.assertIn("72.0 km/h", event["Description"])


class TestMessageTypes(unittest.TestCase):
    """Tests for processing different message types."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.settings = {
            "socket_path": "/tmp/test.sock",
            "camera_1_id": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
            "camera_2_id": "e3e9a385-7fe0-3ba5-5482-a86cde7faf48",
            "ocr_output_name": "Identity:0",
            "nxai_utilities_path": os.path.join(script_dir, "../nxai-utilities/python-utilities")
        }
        self.engine = MagicMock()
    
    @patch('nxai_communication_utils.SocketListener')
    def test_main_processes_detector_message(self, mock_listener_cls):
        """Test processing DetectorMessage with OCR sends response."""
        mock_server = mock_listener_cls.return_value
        mock_conn = MagicMock()
        
        message = DetectorMessage({
            "DeviceID": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
            "Timestamp": datetime.now().timestamp(),
            "BBoxes_xyxy": {"lp": [0, 0, 10, 10]},
            "ObjectsMetaData": {}
        })
        
        mock_server.accept.side_effect = [
            (mock_conn, b"raw_data"),
            KeyboardInterrupt
        ]
        
        with patch.object(module, 'create_anpr_message_from_bytes', return_value=message):
            with patch.object(message, 'handle') as mock_handle:
                with patch.object(SpeedDetectorMessage, 'to_bytes', return_value=b"processed"):
                    with self.assertRaises(KeyboardInterrupt):
                        module.main(self.settings, self.engine)
                    mock_handle.assert_called_once()
                    mock_conn.send.assert_called_once_with(b"processed")
                    mock_conn.close.assert_called()

    @patch('nxai_communication_utils.SocketListener')
    def test_main_calls_handle_for_all_message_types(self, mock_listener_cls):
        """Test that handle() is called for all message types to process OCR."""
        mock_server = mock_listener_cls.return_value
        mock_conn = MagicMock()
        
        # Test with DetectorMessage
        detector_message = DetectorMessage({
            "DeviceID": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
            "Timestamp": datetime.now().timestamp(),
            "BBoxes_xyxy": {"lp": [0, 0, 10, 10]},
            "ObjectsMetaData": {}
        })
        
        mock_server.accept.side_effect = [
            (mock_conn, b"raw_data"),
            KeyboardInterrupt
        ]
        
        with patch.object(module, 'create_anpr_message_from_bytes', return_value=detector_message):
            with patch.object(detector_message, 'handle') as mock_handle:
                with patch.object(SpeedDetectorMessage, 'to_bytes', return_value=b"processed"):
                    with self.assertRaises(KeyboardInterrupt):
                        module.main(self.settings, self.engine)
                    mock_handle.assert_called_once()
                    mock_conn.send.assert_called_once_with(b"processed")
    
    @patch('nxai_communication_utils.SocketListener')
    def test_main_processes_ocr_message(self, mock_listener_cls):
        """Test processing OcrMessage sends response."""
        mock_server = mock_listener_cls.return_value
        mock_conn = MagicMock()
        
        message = CctOcrMessage({
            "DeviceID": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
            "Timestamp": datetime.now().timestamp(),
            "OriginalObjectID": "obj-1",
            "BinaryOutputs": []
        })
        
        mock_server.accept.side_effect = [
            (mock_conn, b"raw_data"),
            KeyboardInterrupt
        ]
        
        with patch.object(module, 'create_anpr_message_from_bytes', return_value=message):
            with patch.object(message, 'handle') as mock_handle:
                with patch.object(message, 'to_bytes', return_value=b"processed"):
                    with self.assertRaises(KeyboardInterrupt):
                        module.main(self.settings, self.engine)
                    
                    mock_handle.assert_called_once()
                    # Verify response is sent for OcrMessage
                    mock_conn.send.assert_called_once_with(b"processed")
                    mock_conn.close.assert_called()
    
    @patch('nxai_communication_utils.SocketListener')
    def test_main_processes_generic_message(self, mock_listener_cls):
        """Test processing GenericMessage sends response."""
        mock_server = mock_listener_cls.return_value
        mock_conn = MagicMock()
        
        message = GenericMessage({
            "DeviceID": "3645c7ee-ca91-e579-e753-1d85af1fd08c",
            "Timestamp": datetime.now().timestamp()
        })
        
        mock_server.accept.side_effect = [
            (mock_conn, b"raw_data"),
            KeyboardInterrupt
        ]
        
        with patch.object(module, 'create_anpr_message_from_bytes', return_value=message):
            with patch.object(message, 'handle') as mock_handle:
                with patch.object(message, 'to_bytes', return_value=b"processed"):
                    with self.assertRaises(KeyboardInterrupt):
                        module.main(self.settings, self.engine)
                    
                    mock_handle.assert_called_once()
                    # Verify response is sent for GenericMessage
                    mock_conn.send.assert_called_once_with(b"processed")
                    mock_conn.close.assert_called()


if __name__ == '__main__':
    unittest.main()
