#!/usr/bin/env python3
"""
Postprocessor Python ANPR Example

This example postprocessor receives OCR results from the CCT model (NxM float32 logits)
and converts them to readable text by finding argmax for each character position.
"""

import os
import sys
import logging
import tempfile
import threading
import configparser
import queue
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional
from pprint import pformat
from abc import ABC, abstractmethod
import msgpack
import numpy as np


logger = logging.getLogger(__name__)


class InferenceMessage(ABC):
    """Represents a generic inference result message from AI Manager."""

    def __init__(self, message: dict):
        self._message = message or {}

    @property
    def original_object_id(self):
        return self._message.get("OriginalObjectID")

    @original_object_id.setter
    def original_object_id(self, value):
        self._message["OriginalObjectID"] = value

    @property
    def binary_outputs(self):
        if "BinaryOutputs" not in self._message or self._message["BinaryOutputs"] is None:
            self._message["BinaryOutputs"] = []
        return self._message["BinaryOutputs"]

    @property
    def bboxes_xyxy(self):
        if "BBoxes_xyxy" not in self._message or self._message["BBoxes_xyxy"] is None:
            self._message["BBoxes_xyxy"] = {}
        return self._message["BBoxes_xyxy"]

    @property
    def objects_metadata(self):
        if "ObjectsMetaData" not in self._message or self._message["ObjectsMetaData"] is None:
            self._message["ObjectsMetaData"] = {}
        return self._message["ObjectsMetaData"]

    @property
    def inference_data(self):
        return self._message.get("InferenceData")

    @staticmethod
    def _parse_payload(data):
        try:
            msg = msgpack.unpackb(data, raw=False, strict_map_key=False)
        except Exception as e:
            logger.error("Failed to unpack message: %s", e)
            raise
        logger.debug("Received message: %s", pformat(msg))
        return msg

    @classmethod
    def create_from_bytes(cls, data):
        """Factory: parse bytes and return a typed message instance."""
        msg = cls._parse_payload(data)
        is_ocr = msg.get("OriginalObjectID") is not None and bool(
            msg.get("BinaryOutputs", [])
        )
        is_detector = bool(msg.get("BBoxes_xyxy"))
        if is_ocr:
            return CctOcrMessage(msg)
        if is_detector:
            return DetectorMessage(msg)
        return GenericMessage(msg)

    @abstractmethod
    def handle(self, ocr_cache) -> None:
        """Process the message and mutate object for response."""
        pass

    @property
    def object_ids(self) -> list[str]:
        ids: list[str] = []
        for meta in (self.objects_metadata or {}).values():
            ids.extend(meta.get("ObjectIDs", []))
        return ids

    def to_bytes(self):
        """Serialize to MessagePack"""
        logger.debug("Sending message: %s", pformat(self._message))
        return msgpack.packb(self._message, use_bin_type=True)

    def get_binary_output(self, name: str):
        for output in self.binary_outputs:
            if output.get("Name") == name:
                return output
        return None

    def get_binary_output_f32(self, name: str):
        output = self.get_binary_output(name)
        if not output:
            return None
        data = output.get("Data")
        if isinstance(data, list):
            data = bytes(data)
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError(f"BinaryOutputs[{name}].Data is not bytes")
        return np.frombuffer(data, dtype=np.float32)


class GenericMessage(InferenceMessage):
    """Fallback for messages that are neither detector nor OCR."""

    def __init__(self, message: dict):
        super().__init__(message)

    def handle(self, ocr_cache) -> None:
        logger.debug("Unknown message type, returning unchanged")
        return None


class DetectorMessage(InferenceMessage):
    """Message from detector model (with bounding boxes)."""

    def __init__(self, message: dict):
        super().__init__(message)

    def to_bytes(self):
        msg = self._message
        bboxes = self.bboxes_xyxy
        meta = self.objects_metadata
        for class_name, coords in bboxes.items():
            entry = meta.get(class_name, {})
            ids = entry.get("ObjectIDs", [])
            keys = entry.get("AttributeKeys", [])
            vals = entry.get("AttributeValues", [])
            num_boxes = len(coords) // 4
            if len(ids) != num_boxes:
                ids = list(ids[:num_boxes]) + [None] * (num_boxes - len(ids))
            while len(keys) < num_boxes:
                keys.append([])
            while len(vals) < num_boxes:
                vals.append([])
            meta[class_name] = {
                "ObjectIDs": ids,
                "AttributeKeys": keys,
                "AttributeValues": vals,
            }
        logger.debug("Sending message: %s", pformat(msg))
        return msgpack.packb(msg, use_bin_type=True)

    def handle(self, ocr_cache) -> None:
        for object_id in set(self.object_ids):
            cached = ocr_cache.get_cached_result(object_id)
            if cached:
                text, conf = cached
                self.update_metadata(object_id, text, conf)
        logger.debug("Returning detector message with cached OCR results")
        return None

    def update_metadata(self, object_id: str, recognized_text: str, confidence: float) -> bool:
        if object_id is None:
            return False
        meta_map = self.objects_metadata
        for class_name, meta in meta_map.items():
            ids = meta.get("ObjectIDs", [])
            if object_id not in ids:
                continue
            idx = ids.index(object_id)
            keys = meta.setdefault("AttributeKeys", [])
            vals = meta.setdefault("AttributeValues", [])
            while len(keys) <= idx:
                keys.append([])
            while len(vals) <= idx:
                vals.append([])
            if not isinstance(keys[idx], list):
                keys[idx] = list(keys[idx]) if keys[idx] else []
            if not isinstance(vals[idx], list):
                vals[idx] = list(vals[idx]) if vals[idx] else []

            # Update License Plate Text
            if "License Plate Text" in keys[idx]:
                k_idx = keys[idx].index("License Plate Text")
                if len(vals[idx]) > k_idx:
                    vals[idx][k_idx] = recognized_text
                else:
                    vals[idx].append(recognized_text)
            else:
                keys[idx].append("License Plate Text")
                vals[idx].append(recognized_text)

            # Update Confidence (formatted to 4 decimal places)
            confidence_str = f"{float(confidence):.4f}"
            if "Confidence" in keys[idx]:
                c_idx = keys[idx].index("Confidence")
                if len(vals[idx]) > c_idx:
                    vals[idx][c_idx] = confidence_str
                else:
                    vals[idx].append(confidence_str)
            else:
                keys[idx].append("Confidence")
                vals[idx].append(confidence_str)

            return True
        return False


class OcrMessage(InferenceMessage, ABC):
    """Message from OCR model (with BinaryOutputs)."""

    def __init__(self, message: dict):
        super().__init__(message)

    @abstractmethod
    def decode_ocr_logits(self, engine, output_name):
        """Decode OCR logits and return recognized text."""
        pass

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


    def extract_logits_array(self, output_name):
        """Extract logits array from message."""
        pass


class CctOcrMessage(OcrMessage):
    """OCR message for CCT model (NxM logits)."""

    def __init__(self, message: dict, expected_logits_shape=(9, 37)):
        self.expected_logits_shape = expected_logits_shape  # (N characters, M classes)
        super().__init__(message)

    def decode_ocr_logits(self, engine, output_name):
        """
        Decode OCR inference results from CCT model.
        Expected input shape is defined by expected_logits_shape.
        """
        ocr_logits = self.extract_logits_array(output_name)
        return engine.decode_logits(ocr_logits)

    def extract_logits_array(self, output_name):
        ocr_logits = None
        binary_logits = self.get_binary_output_f32(output_name)
        expected_size = self.expected_logits_shape[0] * self.expected_logits_shape[1]
        if binary_logits is None:
            ocr_logits = None
        elif binary_logits.size != expected_size:
            logger.warning(
                "BinaryOutputs %s size %d does not match expected %s",
                output_name, binary_logits.size, self.expected_logits_shape
            )
        else:
            ocr_logits = binary_logits.reshape(self.expected_logits_shape)
        if ocr_logits is None and self.inference_data is not None:
            logger.info("InferenceData type: %s", type(self.inference_data))
            logger.info(
                "InferenceData shape: %s", getattr(self.inference_data, 'shape', 'no shape')
            )
            if isinstance(self.inference_data, list):
                ocr_logits = np.array(self.inference_data)
            elif hasattr(self.inference_data, 'shape'):
                ocr_logits = np.array(self.inference_data)
            else:
                logger.warning("Unexpected InferenceData format: %s", self.inference_data)
        return ocr_logits


class LogitsOcrEngine:

    def __init__(self, expected_logits_shape, char_map: str):
        self._expected_logits_shape = expected_logits_shape
        self._char_map = char_map
        self._validate_config()

    def _validate_config(self):
        if (
            not isinstance(self._expected_logits_shape, tuple)
            or len(self._expected_logits_shape) != 2
        ):
            raise ValueError(
                f"expected_logits_shape must be (rows, cols), got {self._expected_logits_shape}"
            )
        if self._expected_logits_shape[0] <= 0 or self._expected_logits_shape[1] <= 0:
            raise ValueError(
                f"expected_logits_shape must be positive, got {self._expected_logits_shape}"
            )
        if not isinstance(self._char_map, str) or not self._char_map:
            raise ValueError("char_map must be a non-empty string")
        if len(self._char_map) != self._expected_logits_shape[1]:
            raise ValueError(
                f"char_map length {len(self._char_map)} does not match "
                f"expected_logits_shape[1] {self._expected_logits_shape[1]}"
            )

    def apply(self, processed_image) -> str:
        return self.decode_logits(np.array(processed_image)) or ""

    def decode_logits(self, ocr_logits):
        if ocr_logits is None:
            logger.warning("No OCR logits found in BinaryOutputs or InferenceData")
            return None
        logger.debug("OCR logits shape: %s", ocr_logits.shape)
        logger.debug("OCR logits dtype: %s", ocr_logits.dtype)
        logger.debug("OCR logits sample: %s", ocr_logits.flatten()[:20])
        if ocr_logits.shape != self._expected_logits_shape:
            logger.warning(
                "Unexpected OCR logits shape: %s, expected %s",
                ocr_logits.shape, self._expected_logits_shape
            )
            return None
        recognized_text = ""
        confidences = []
        for i in range(self._expected_logits_shape[0]):
            char_logits = ocr_logits[i]
            char_index = np.argmax(char_logits)
            confidence = char_logits[char_index]

            if char_index < len(self._char_map):
                char = self._char_map[char_index]
            else:
                char = "?"

            recognized_text += char
            confidences.append(float(confidence))

            logger.debug(
                "Position %d: '%s' (index %d, confidence %.4f)",
                i, char, char_index, confidence
            )
        recognized_text = recognized_text.rstrip()
        min_confidence = min(confidences) if confidences else 0.0
        logger.debug("Recognized text: '%s'", recognized_text)
        logger.debug("Confidences: %s", confidences)
        return recognized_text, min_confidence


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


def _config():
    """Read configuration from optional INI at fixed path (../etc/plugin.anpr.ini)."""
    script_location = os.path.dirname(sys.argv[0])
    config_file = os.path.join(script_location, "..", "etc", "plugin.anpr.ini")
    temp_dir = Path(tempfile.gettempdir())
    settings = {
        "log_level": "DEBUG",
        "ocr_worker_count": max(1, min(4, os.cpu_count() or 1)),
        "socket_path": str(temp_dir / "postprocessor-anpr-example.sock"),
        "log_file": str(temp_dir / "postprocessor-anpr-example.log"),
        "ocr_output_name": "Identity:0",
        "nxai_utilities_path": os.path.join(script_location, "..", "nxai-utilities", "python-utilities"),
    }
    if not os.path.exists(config_file):
        logger.info("Configuration file %s not found, using default settings", config_file)
        return settings
    configuration = configparser.ConfigParser()
    logger.info("Reading configuration from: %s", config_file)
    try:
        configuration.read(config_file)
    except Exception as e:
        logger.error("Failed to read configuration file %s: %s. Using default settings.", config_file, e)
        return settings
    if "common" in configuration:
        settings["log_level"] = configuration.get("common", "log_level", fallback=settings["log_level"])
        settings["socket_path"] = configuration.get("common", "socket_path", fallback=settings["socket_path"])
        settings["log_file"] = configuration.get("common", "log_file", fallback=settings["log_file"])
        settings["nxai_utilities_path"] = configuration.get("common", "nxai_utilities_path", fallback=settings["nxai_utilities_path"])
    if "ocr" in configuration:
        settings["ocr_worker_count"] = configuration.getint(
            "ocr", "worker_count", fallback=settings["ocr_worker_count"]
        )
        settings["ocr_output_name"] = configuration.get(
            "ocr", "output_name", fallback=settings["ocr_output_name"]
        )
    logger.debug("Read configuration done")
    return settings


def _setup_logging(level_str: str, log_file: str):
    """Set up logging configuration"""
    numeric_level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - anpr-example - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w")
        ],
        force=True
    )
    logger.info("Logging initialized at level %s", level_str)


def main(settings, engine, ocr_pool=None):
    """Main postprocessor loop"""
    logger.info("=== STARTING ANPR POSTPROCESSOR ===")
    logger.info("Socket path: %s", settings["socket_path"])
    logger.info("Current working directory: %s", os.getcwd())
    server = nxai_communication_utils.SocketListener(settings["socket_path"])
    ocr_cache = OcrCache(engine, settings["ocr_output_name"], ocr_pool)
    while True:
        # Wait for input message from runtime
        logger.debug("Waiting for input message")
        connection = None
        try:
            connection, input_message = server.accept()
            logger.debug("Received input message")
        except nxai_communication_utils.SocketTimeout:
            # Request timed out. Continue waiting
            continue
        try:
            message = InferenceMessage.create_from_bytes(input_message)
            logger.debug("Processing message: %s", message.__class__.__name__)
            message.handle(ocr_cache)
            try:
                connection.send(message.to_bytes())
            except Exception as e:
                logger.warning("Failed to send response: %s", e)
        except Exception as e:
            logger.error("Error processing message: %s", e, exc_info=True)
            try:
                # Return original message on error
                connection.send(input_message)
            except Exception:
                pass
        finally:
            if connection is not None:
                connection.close()


if __name__ == "__main__":
    settings = _config()

    _setup_logging(settings["log_level"], settings["log_file"])
    if len(sys.argv) > 1:
        settings["socket_path"] = sys.argv[1]
    logger.debug("Input parameters: %s", sys.argv)
    logger.info("Configuration loaded:")
    for key, val in settings.items():
        logger.info("  %s = %s", key, val)

    if settings["nxai_utilities_path"] not in sys.path:
        sys.path.append(settings["nxai_utilities_path"])

    import nxai_communication_utils
    logger.info(
        "nxai_communication_utils loaded from %s", nxai_communication_utils.__file__)
    try:
        engine = LogitsOcrEngine(
            expected_logits_shape=(9, 37),
            char_map="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ",
        )
        pool = OcrWorkerPool(engine, settings["ocr_worker_count"])
        main(settings, engine, pool)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        sys.exit(1)
    finally:
        if "pool" in locals():
            pool.stop()
