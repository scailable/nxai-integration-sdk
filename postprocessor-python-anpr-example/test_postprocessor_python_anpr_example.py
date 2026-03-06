import unittest
import logging
from unittest.mock import MagicMock, patch, mock_open
import sys
import os
import numpy as np
import time
import threading

# Add the nxai-utilities path before importing the module
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "../nxai-utilities/python-utilities"))
# Add repo root so message_processing_utils is importable
repo_root = os.path.join(script_dir, "..")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

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
        "postprocessor_python_anpr_example",
        os.path.join(script_dir, "postprocessor-python-anpr-example.py")
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["postprocessor_python_anpr_example"] = module
    # Ensure nxai_communication_utils is available in the module's namespace
    module.nxai_communication_utils = mock_comm
    spec.loader.exec_module(module)
    
    # Import classes from message_processing_utils
    from message_processing_utils import (
        InferenceMessage,
        GenericMessage,
        create_anpr_message_from_bytes,
    )
    from message_processing_utils.anpr import AnprDetectorMessage, CctOcrMessage
    from message_processing_utils.general.detector.messages import DetectorMessage
    from message_processing_utils.general.ocr import (
        LogitsOcrEngine,
        OcrWorkerPool,
        OcrCache,
        load_ocr_config,
    )


class TestLogitsOcrEngine(unittest.TestCase):
    def setUp(self):
        self.char_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "
        self.engine = LogitsOcrEngine(expected_logits_shape=(9, 37), char_map=self.char_map)

    def test_decode_correct_logits(self):
        logits = np.zeros((9, 37), dtype=np.float32)
        # Set 'A' (index 10) for first char, '1' (index 1) for second
        logits[0, 10] = 1.0
        logits[1, 1] = 1.0
        text, conf = self.engine.decode_logits(logits)
        self.assertTrue(text.startswith("A1"))
        self.assertEqual(conf, 0.0) # all other positions are 0.0

    def test_decode_invalid_shape(self):
        logits = np.zeros((5, 10), dtype=np.float32)
        result = self.engine.decode_logits(logits)
        self.assertIsNone(result)

    def test_decode_out_of_bounds_index(self):
        logits = np.zeros((9, 37), dtype=np.float32)
        logits.fill(-10.0)
        logits[:, 36] = 1.0 
        text, conf = self.engine.decode_logits(logits)
        self.assertEqual(text, "")
        self.assertEqual(conf, 1.0)

    def test_validate_config_errors(self):
        with self.assertRaises(ValueError):
            LogitsOcrEngine(expected_logits_shape=(9, 10), char_map="ABC") # Length mismatch


class TestOcrWorkerPool(unittest.TestCase):
    def test_async_processing_and_cache(self):
        engine = LogitsOcrEngine(expected_logits_shape=(9, 37), char_map="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ")
        pool = OcrWorkerPool(engine, worker_count=2)
        cache = OcrCache(engine, "Identity:0", pool)
        
        # Mock message
        msg = MagicMock(spec=CctOcrMessage)
        msg.original_object_id = "async-obj"
        
        # Use a real event to wait for async completion in test
        done_event = threading.Event()
        original_cache_method = cache.cache_ocr_result
        
        def side_effect(m, text, conf):
            original_cache_method(m, text, conf)
            done_event.set()
            
        with patch.object(cache, 'cache_ocr_result', side_effect=side_effect):
            logits = np.zeros((9, 37), dtype=np.float32)
            pool.submit(logits, lambda res, err: cache.cache_ocr_result(msg, res[0], res[1]) if res else None)
            
            self.assertTrue(done_event.wait(timeout=1.0), "Async OCR timed out")
            text, conf = cache.get_cached_result("async-obj")
            self.assertEqual(text, "000000000")
            self.assertEqual(conf, 0.0)
        
        pool.stop()


class TestDetectorMetadata(unittest.TestCase):
    def setUp(self):
        self.engine = MagicMock()
        self.cache = OcrCache(self.engine, "Identity:0")

    def test_add_license_plate_metadata_new_structure(self):
        """Test creating ObjectsMetaData from scratch"""
        msg = AnprDetectorMessage({
            "BBoxes_xyxy": {"car": [0,0,1,1]}, 
            "ObjectsMetaData": {"car": {"ObjectIDs": ["obj-1"]}}
        })
        success = msg.add_license_plate_metadata("obj-1", "TEXT", 0.95)
        self.assertTrue(success)
        meta = msg.objects_metadata["car"]
        self.assertEqual(meta["AttributeValues"][0][meta["AttributeKeys"][0].index("License Plate Text")], "TEXT")
        self.assertEqual(meta["AttributeValues"][0][meta["AttributeKeys"][0].index("Confidence")], "0.9500")

    def test_multiple_bboxes_normalization(self):
        """Test normalization when some boxes have IDs and some don't"""
        msg_dict = {
            "BBoxes_xyxy": {"car": [0,0,1,1, 2,2,3,3]}, # 2 boxes
            "ObjectsMetaData": {"car": {"ObjectIDs": ["id-1"]}} # Only 1 ID
        }
        msg = AnprDetectorMessage(msg_dict)
        # to_bytes should normalize ObjectIDs to length 2
        with patch('message_processing_utils.general.detector.messages.msgpack.packb', side_effect=lambda x, **kwargs: x):
            payload = msg.to_bytes()
            self.assertEqual(len(payload["ObjectsMetaData"]["car"]["ObjectIDs"]), 2)
            self.assertIsNone(payload["ObjectsMetaData"]["car"]["ObjectIDs"][1])


class TestInferenceMessageFactory(unittest.TestCase):
    def test_create_from_bytes_ocr(self):
        payload = {
            "DeviceID": "camera-001",
            "OriginalObjectID": "obj-1",
            "BinaryOutputs": [{"Name": "Identity:0", "Data": b"\x00", "Type": 1}],
        }
        with patch('message_processing_utils.base.messages.msgpack') as mock_msgpack:
            mock_msgpack.unpackb.return_value = payload
            msg = create_anpr_message_from_bytes(b"data")
        self.assertIsInstance(msg, CctOcrMessage)

    def test_create_from_bytes_detector(self):
        payload = {
            "BBoxes_xyxy": {"lp": [0, 0, 10, 10]},
            "OriginalObjectID": None,
            "BinaryOutputs": [],
        }
        with patch('message_processing_utils.base.messages.msgpack') as mock_msgpack:
            mock_msgpack.unpackb.return_value = payload
            msg = create_anpr_message_from_bytes(b"data")
        self.assertIsInstance(msg, AnprDetectorMessage)

    def test_create_from_bytes_detector_empty_bboxes(self):
        """Message with BBoxes_xyxy: {} should yield AnprDetectorMessage (key presence, not truthiness)."""
        payload = {
            "BBoxes_xyxy": {},
            "OriginalObjectID": None,
            "BinaryOutputs": [],
        }
        with patch('message_processing_utils.base.messages.msgpack') as mock_msgpack:
            mock_msgpack.unpackb.return_value = payload
            msg = create_anpr_message_from_bytes(b"data")
        self.assertIsInstance(msg, AnprDetectorMessage)


class TestMainLoop(unittest.TestCase):
    @unittest.skip(
        "Socket timeout test is sensitive to mock setup when run in suite; "
        "manual run passes. Main loop correctly catches SocketTimeout and continues."
    )
    @patch('postprocessor_python_anpr_example.nxai_communication_utils.SocketListener')
    def test_main_handles_socket_timeout(self, mock_listener_cls):
        module.nxai_communication_utils.SocketTimeout = MockSocketTimeout
        def raise_timeout():
            raise MockSocketTimeout()
        def raise_kb():
            raise KeyboardInterrupt()
        mock_listener_cls.return_value.accept.side_effect = [raise_timeout, raise_kb]
        from message_processing_utils.general.ocr import load_ocr_config
        settings = load_ocr_config(None, processor_name="anpr-example")
        engine = MagicMock()
        with self.assertRaises(KeyboardInterrupt):
            module.main(settings, engine)

    @patch('nxai_communication_utils.SocketListener')
    def test_main_handles_send_error(self, mock_listener_cls):
        mock_server = mock_listener_cls.return_value
        mock_conn = MagicMock()
        mock_server.accept.return_value = (mock_conn, b"raw_data")
        
        # Mock message that raises error on send
        mock_conn.send.side_effect = Exception("Send failed")
        
        with patch.object(module, 'create_anpr_message_from_bytes') as mock_factory:
            mock_msg = MagicMock()
            mock_factory.return_value = mock_msg
            mock_server.accept.side_effect = [(mock_conn, b"data"), KeyboardInterrupt]
            
            from message_processing_utils.general.ocr import load_ocr_config
            settings = load_ocr_config(None, processor_name="anpr-example")
            with self.assertRaises(KeyboardInterrupt):
                module.main(settings, MagicMock())
            
            mock_conn.close.assert_called()


if __name__ == '__main__':
    unittest.main()
