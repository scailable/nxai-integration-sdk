#!/usr/bin/env python3
"""
Postprocessor Python Measure Average Car Speed

This postprocessor extends the stitch OCR result functionality to measure
average car speed by tracking license plates across two cameras.
"""

import os
import sys
import logging
import configparser
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

# Add repo root to path so message_processing_utils is importable
script_location = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.join(script_location, "..")
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from config_utils import (
    setup_logging,
    get_nxai_utilities_library_path,
    get_postprocessor_base_dir,
)
from message_processing_utils import (
    InferenceMessage,
    create_anpr_message_from_bytes,
)
from message_processing_utils.general.detector import DetectorMessage
from message_processing_utils.anpr import SpeedDetectorMessage
from message_processing_utils.general.ocr import (
    LogitsOcrEngine,
    OcrWorkerPool,
    OcrCache,
    load_ocr_config,
)
from speed_cache import SpeedMeasurementCache

logger = logging.getLogger(__name__)

# Prefix for UI setting names in ExternalProcessorSettings (must match external_postprocessors.json)
UI_SETTINGS_PREFIX = "externalprocessor."


def _merge_speed_settings(
    settings: Dict[str, Any],
    ui_settings: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Merge INI/startup settings with UI settings from current message.
    UI overrides when key is present and value is valid.
    Returns dict with camera_1_id, camera_2_id, distance_between_cameras_m,
    second_appearance_timeout_sec.
    """
    result = {
        "camera_1_id": settings.get("camera_1_id", "").strip(),
        "camera_2_id": settings.get("camera_2_id", "").strip(),
        "distance_between_cameras_m": float(settings.get("distance_between_cameras_m", 100.0)),
        "second_appearance_timeout_sec": float(
            settings.get("second_appearance_timeout_sec", 60.0)
        ),
    }
    if not isinstance(ui_settings, dict):
        return result
    key_cam1 = UI_SETTINGS_PREFIX + "camera_1_id"
    key_cam2 = UI_SETTINGS_PREFIX + "camera_2_id"
    key_dist = UI_SETTINGS_PREFIX + "distance_between_cameras_m"
    key_timeout = UI_SETTINGS_PREFIX + "second_appearance_timeout_sec"
    if key_cam1 in ui_settings and ui_settings[key_cam1] is not None:
        val = str(ui_settings[key_cam1]).strip()
        if val:
            result["camera_1_id"] = val
    if key_cam2 in ui_settings and ui_settings[key_cam2] is not None:
        val = str(ui_settings[key_cam2]).strip()
        if val:
            result["camera_2_id"] = val
    if key_dist in ui_settings and ui_settings[key_dist] is not None:
        try:
            result["distance_between_cameras_m"] = float(ui_settings[key_dist])
        except (TypeError, ValueError):
            pass
    if key_timeout in ui_settings and ui_settings[key_timeout] is not None:
        try:
            result["second_appearance_timeout_sec"] = float(ui_settings[key_timeout])
        except (TypeError, ValueError):
            pass
    return result


def _extract_license_plates_from_message(
        message: DetectorMessage) -> List[Tuple[str, str, str, datetime]]:
    """
    Extract recognized license plates from processed DetectorMessage.
    
    Args:
        message: Processed DetectorMessage with OCR results.
    
    Returns:
        List of tuples (object_id, license_plate, device_id, timestamp).
    """
    if not message.timestamp:
        logger.debug("No timestamp in message, skipping speed measurement")
        return []
    detection_time = message.timestamp
    if not isinstance(detection_time, datetime):
        logger.warning("Timestamp is not datetime: %s", type(detection_time))
        return []
    device_id = message.device_id
    results = []
    for _class_name, meta in message.objects_metadata.items():
        object_ids = meta.get("ObjectIDs", [])
        attribute_keys = meta.get("AttributeKeys", [])
        attribute_values = meta.get("AttributeValues", [])
        for idx, object_id in enumerate(object_ids):
            if idx >= len(attribute_keys) or idx >= len(attribute_values):
                continue
            keys = attribute_keys[idx]
            values = attribute_values[idx]
            if not isinstance(keys, list) or not isinstance(values, list):
                logger.debug(
                    "Skipping invalid attribute keys/values structure for object %s",
                    object_id
                )
                continue
            try:
                text_idx = keys.index("License Plate Text")
            except ValueError:
                logger.debug(
                    "License Plate Text attribute not found in keys for object %s",
                    object_id
                )
                continue
            if text_idx >= len(values):
                continue
            recognized_text = values[text_idx]
            if not recognized_text or not recognized_text.strip():
                continue
            results.append((
                object_id,
                recognized_text.strip(),
                device_id,
                detection_time
            ))
    
    return results


def main(
    settings: Dict[str, Any],
    engine: Any,
    ocr_pool: Optional[Any] = None,
    speed_cache: Optional[Any] = None,  
) -> None:
    """
    Main postprocessor loop.
    
    Args:
        settings: Configuration dictionary with socket_path, camera IDs, etc.
        engine: OCR engine instance.
        ocr_pool: Optional OCR worker pool.
        speed_cache: Optional speed measurement cache.
        
    """
    logger.info("=== STARTING MEASURE AVERAGE CAR SPEED POSTPROCESSOR ===")
    logger.info("Socket path: %s", settings["socket_path"])
    logger.info("Current working directory: %s", os.getcwd())
    if settings["nxai_utilities_path"] not in sys.path:
        sys.path.append(settings["nxai_utilities_path"])
    import nxai_communication_utils
    lib_path = get_nxai_utilities_library_path()
    if lib_path is not None:
        nxai_communication_utils.initializeLibrary(lib_path)
    server = nxai_communication_utils.SocketListener(settings["socket_path"])
    ocr_cache = OcrCache(engine, settings["ocr_output_name"], ocr_pool)
    while True:
        logger.debug("Waiting for input message")
        connection = None
        try:
            connection, input_message = server.accept()
        except nxai_communication_utils.SocketTimeout:
            logger.debug("Socket timeout, continuing to wait for messages")
            continue
        except Exception as e:
            logger.error("Error accepting connection: %s", e, exc_info=True)
            continue
        logger.debug("Received input message")
        try:
            message = create_anpr_message_from_bytes(input_message)
        except Exception as e:
            logger.error("Error creating message from bytes: %s", e, exc_info=True)
            if connection is not None:
                connection.close()
            continue
        logger.debug("Processing message: %s", message.__class__.__name__)
        ui_settings = message._message.get("ExternalProcessorSettings") or {}
        current = _merge_speed_settings(settings, ui_settings)
        valid_speed_config = bool(
            current["camera_1_id"]
            and current["camera_2_id"]
            and current["distance_between_cameras_m"] > 0
        )
        if valid_speed_config and speed_cache is not None:
            speed_cache.update_config(
                current["camera_1_id"],
                current["camera_2_id"],
                current["distance_between_cameras_m"],
                current["second_appearance_timeout_sec"],
            )
        try:
            message.handle(ocr_cache)
        except Exception as e:
            logger.error("Error handling message with OCR cache: %s", e, exc_info=True)
        if not isinstance(message, DetectorMessage):
            logger.debug(
                "Message is not a DetectorMessage, skipping speed measurement"
            )
        else:
            speed_detector_message = SpeedDetectorMessage(message._message)
            if valid_speed_config and speed_cache is not None:
                logger.debug(
                    "Extracting license plates from DetectorMessage for "
                    "speed measurement"
                )
                license_plates = _extract_license_plates_from_message(speed_detector_message)
                for object_id, license_plate, device_id, detection_time in license_plates:
                    logger.debug(
                        "Adding license plate %s to speed cache (camera: %s)",
                        license_plate, device_id
                    )
                    avg_speed = speed_cache.add_detection(
                        license_plate, device_id, detection_time
                    )
                    if avg_speed is None:
                        continue
                    speed_kmh = avg_speed * 3.6
                    logger.info(
                        "Average speed for license plate %s: %.1f km/h",
                        license_plate, speed_kmh
                    )
                    description = f"License plate {license_plate} detected with average speed {speed_kmh:.1f} km/h"
                    speed_detector_message.add_event("speed.measurement", "Speed Measurement", description)
                    logger.debug("Added speed measurement event for license plate %s", license_plate)
                    speed_detector_message.add_speed_metadata(object_id, avg_speed)
            if valid_speed_config and speed_cache is not None:
                for object_id, license_plate, _, _ in license_plates:
                    speed_ms = speed_cache.get_speed(license_plate)
                    if speed_ms is not None:
                        speed_detector_message.add_speed_metadata(object_id, speed_ms)
                        speed_cache.update_last_seen(license_plate)
                        logger.debug(
                            "Added speed %.1f m/s to object %s (license plate: %s)",
                            speed_ms, object_id, license_plate
                        )
            message = speed_detector_message
        try:
            connection.send(message.to_bytes())
        except Exception as e:
            logger.warning("Failed to send response: %s", e)
        finally:
            if connection is not None:
                connection.close()


def load_speed_measurement_config(config_path: str, settings: Dict[str, Any]) -> None:
    """
    Load speed measurement configuration from INI file.
    
    Args:
        config_path: Path to configuration file.
        settings: Settings dictionary to update with speed measurement config.
    
    Raises:
        RuntimeError: If config file is missing or [speed_measurement] section is absent.
    """
    if not os.path.exists(config_path):
        raise RuntimeError(
            f"Config file not found at {config_path}. "
            "camera_1_id and camera_2_id are required parameters."
        )
    configuration = configparser.ConfigParser()
    configuration.read(config_path)
    if "speed_measurement" not in configuration:
        raise RuntimeError("Missing [speed_measurement] section in config file")
    speed_section = configuration["speed_measurement"]
    camera_1_id = speed_section.get("camera_1_id", "").strip()
    camera_2_id = speed_section.get("camera_2_id", "").strip()
    if not camera_1_id or not camera_2_id:
        logger.warning(
            "Camera IDs not set in INI (camera_1_id=%r, camera_2_id=%r). "
            "Speed will not be calculated until you set them via the plugin UI.",
            camera_1_id or "(empty)",
            camera_2_id or "(empty)",
        )
    settings["camera_1_id"] = camera_1_id
    settings["camera_2_id"] = camera_2_id
    settings["distance_between_cameras_m"] = speed_section.getfloat(
        "distance_between_cameras_m",
        fallback=100.0,
    )
    settings["second_appearance_timeout_sec"] = speed_section.getfloat(
        "second_appearance_timeout_sec",
        fallback=60.0,
    )


if __name__ == "__main__":
    script_location = get_postprocessor_base_dir()
    default_config_path = os.path.join(
        script_location, "..", "etc", "plugin.measure-average-car-speed.ini"
    )
    config_path = default_config_path if os.path.exists(default_config_path) else None
    settings = load_ocr_config(
        config_path, processor_name="measure-average-car-speed"
    )
    if len(sys.argv) > 1:
        settings["socket_path"] = sys.argv[1]
    setup_logging(
        settings["log_level"],
        settings["log_file"],
        processor_name="measure-average-car-speed"
    )
    logger.debug("Input parameters: %s", sys.argv)
    if config_path is not None:
        try:
            load_speed_measurement_config(config_path, settings)
        except RuntimeError as e:
            logger.error("Configuration error: %s", e)
            sys.exit(1)
    else:
        settings["camera_1_id"] = ""
        settings["camera_2_id"] = ""
        settings["distance_between_cameras_m"] = 100.0
        settings["second_appearance_timeout_sec"] = 60.0
        logger.info(
            "No INI config at %s; using defaults. Configure via UI or INI.",
            default_config_path
        )
    logger.info("Configuration loaded:")
    for key, val in settings.items():
        logger.info("  %s = %s", key, val)
    if settings["nxai_utilities_path"] not in sys.path:
        sys.path.append(settings["nxai_utilities_path"])
    import nxai_communication_utils
    logger.info(
        "nxai_communication_utils loaded from %s",
        nxai_communication_utils.__file__
    )
    try:
        engine = LogitsOcrEngine(
            expected_logits_shape=(9, 37),
            char_map="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ ",
        )
    except Exception as e:
        logger.error("Failed to create OCR engine: %s", e, exc_info=True)
        sys.exit(1)
    try:
        pool = OcrWorkerPool(engine, settings["ocr_worker_count"])
    except Exception as e:
        logger.error("Failed to create OCR worker pool: %s", e, exc_info=True)
        sys.exit(1)
    try:
        speed_cache = SpeedMeasurementCache(
            timeout_sec=settings["second_appearance_timeout_sec"],
            distance_m=settings["distance_between_cameras_m"],
            camera_1_id=settings["camera_1_id"],
            camera_2_id=settings["camera_2_id"],
            logger_instance=logger
        )
    except Exception as e:
        logger.error("Failed to create speed cache: %s", e, exc_info=True)
        sys.exit(1)
    try:
        main(settings, engine, pool, speed_cache)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error("Unexpected error: %s", e, exc_info=True)
        sys.exit(1)
    finally:
        if "pool" in locals():
            pool.stop()
        if "speed_cache" in locals():
            speed_cache.stop()