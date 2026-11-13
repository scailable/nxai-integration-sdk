#!/usr/bin/env python3
"""
Postprocessor Python Measure Average Car Speed (v5 SDK)

Receives OCR results from a chained CCT model (the
same wire shape as postprocessor-python-anpr-example) and decodes them into
readable license-plate text via message_processing_utils.anpr, then tracks
matching plates seen on two configured cameras to compute an average speed
over the distance between them. Attaches a "Speed" attribute to the plate
object on subsequent frames and emits a 'speed.measurement' event once a
plate has cleared both cameras.

Behavior change from v4:
- camera_1_id, camera_2_id, distance_between_cameras_m and
  second_appearance_timeout_sec are now live UserSettings controls,
  read fresh from each message with hardcoded defaults,
  instead of an INI file value overridden by an `externalprocessor.`-prefixed
  ExternalProcessorSettings key.
- OCR worker pool size, OCR output tensor name, and cache TTL are launch-time
  Config values (`--key value` in external_postprocessors.json), replacing
  the removed `plugin.measure-average-car-speed.ini` file.
- The per-socket-path singleton lock is removed: v5 launches exactly one
  process per registration entry over its own dedicated IPC channels, so
  there is no shared socket path to race on.
- The emitted event drops the v4 Caption field (v5 EventMetadata carries
  only type_id + description) and carries the id 'speed.measurement' exactly
  as registered in external_postprocessors.json, with no 'nx.nxai.' prefix.
"""

import os
import sys
import time
import logging

script_location = os.path.dirname(os.path.abspath(__file__))
if os.path.join(script_location, "..") not in sys.path:
    sys.path.insert(0, os.path.join(script_location, ".."))

import nxai_postprocessor_sdk
from message_processing_utils import create_speed_message_from_bytes
from message_processing_utils.anpr import SpeedDetectorMessage
from message_processing_utils.general.ocr import LogitsOcrEngine, OcrWorkerPool, OcrCache
from speed_cache import SpeedMeasurementCache

logger = logging.getLogger(__name__)

PROCESSOR_NAME = "measure-average-car-speed"
EVENT_ID_SPEED_MEASUREMENT = "speed.measurement"

DEFAULT_CAMERA_1_ID = ""
DEFAULT_CAMERA_2_ID = ""
DEFAULT_DISTANCE_BETWEEN_CAMERAS_M = 100.0
DEFAULT_SECOND_APPEARANCE_TIMEOUT_SEC = 60.0
DEFAULT_OCR_WORKER_COUNT = 2
DEFAULT_OCR_OUTPUT_NAME = "Identity:0"
DEFAULT_CACHE_TTL_SEC = 300.0
EXPECTED_LOGITS_SHAPE = (9, 37)
CHAR_MAP = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "

# ============================================================================
# Business Logic
# ============================================================================


def read_camera_config(user_settings):
    """Read this message's camera/distance/timeout settings, defaulting to
    "unconfigured" values that disable speed measurement until set via the UI.
    """
    return {
        "camera_1_id": str(user_settings.get("camera_1_id", DEFAULT_CAMERA_1_ID)).strip(),
        "camera_2_id": str(user_settings.get("camera_2_id", DEFAULT_CAMERA_2_ID)).strip(),
        "distance_between_cameras_m": _get_float_setting(
            user_settings, "distance_between_cameras_m", DEFAULT_DISTANCE_BETWEEN_CAMERAS_M
        ),
        "second_appearance_timeout_sec": _get_float_setting(
            user_settings,
            "second_appearance_timeout_sec",
            DEFAULT_SECOND_APPEARANCE_TIMEOUT_SEC,
        ),
    }


def handle_message(message, ocr_cache, speed_cache, camera_config):
    """Cache/decode this message's OCR contribution and, for detector
    messages, track matching plates across the two configured cameras.

    Args:
        message: SpeedDetectorMessage or CctOcrMessage from create_speed_message_from_bytes.
        ocr_cache: OcrCache shared across messages, keyed by original_object_id.
        speed_cache: SpeedMeasurementCache shared across messages.
        camera_config: This message's camera_1_id/camera_2_id/
            distance_between_cameras_m/second_appearance_timeout_sec, from
            read_camera_config().

    Returns:
        The message, with speed attributes and events applied when it was a
        detector message.
    """
    message.handle(ocr_cache)
    if not isinstance(message, SpeedDetectorMessage):
        return message

    valid_config = bool(
        camera_config["camera_1_id"]
        and camera_config["camera_2_id"]
        and camera_config["distance_between_cameras_m"] > 0
        and camera_config["second_appearance_timeout_sec"] > 0
    )
    if not valid_config:
        return message

    detection_time = message.timestamp
    if detection_time is None:
        logger.debug("No timestamp in message, skipping speed measurement")
        return message

    speed_cache.update_config(
        camera_config["camera_1_id"],
        camera_config["camera_2_id"],
        camera_config["distance_between_cameras_m"],
        camera_config["second_appearance_timeout_sec"],
    )
    device_id = message.device_id
    for object_id, license_plate in _extract_license_plates(message):
        avg_speed = speed_cache.add_detection(license_plate, device_id, detection_time)
        if avg_speed is not None:
            speed_kmh = avg_speed * 3.6
            description = (
                f"License plate {license_plate} detected with average speed "
                f"{speed_kmh:.1f} km/h"
            )
            message.add_event(EVENT_ID_SPEED_MEASUREMENT, description)
            message.add_speed_metadata(object_id, avg_speed)
            logger.info(
                "Average speed for license plate %s: %.1f km/h", license_plate, speed_kmh
            )
            continue
        speed_ms = speed_cache.get_speed(license_plate)
        if speed_ms is not None:
            message.add_speed_metadata(object_id, speed_ms)
            speed_cache.update_last_seen(license_plate)
    return message


def run_message_loop(processor, ocr_cache, speed_cache):
    """Serve messages until the engine stops sending or the response channel fails.

    A failure while handling one message must not end the loop: the engine keeps
    sending frames, and a processor that exits is dropped from the pipeline.

    Args:
        processor: An initialized ExternalPostprocessor.
        ocr_cache: OcrCache shared across messages.
        speed_cache: SpeedMeasurementCache shared across messages.
    """
    while True:
        logger.debug("Waiting for input message")
        message = None
        try:
            ocr_cache.cleanup(time.monotonic())

            message_bytes = processor.wait_for_message(timeout_ms=5000)
            if message_bytes is None:
                if processor.exit_requested:
                    logger.info("Received graceful exit signal, shutting down")
                    return
                logger.debug("Timeout waiting for message")
                continue

            logger.debug(f"Received message ({len(message_bytes)} bytes)")

            message = create_speed_message_from_bytes(message_bytes)
            if message is None:
                logger.warning("Could not parse message, skipping")
                continue

            camera_config = read_camera_config(message.user_settings)
            message = handle_message(message, ocr_cache, speed_cache, camera_config)

            if not processor.send_response(message.to_bytes()):
                logger.error("Failed to send response")
                return

            logger.debug("Response sent successfully")
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            # The engine expects a response for every frame it sent, so answer
            # with the message as far as it got rather than dropping the frame.
            if message is not None:
                try:
                    if not processor.send_response(message.to_bytes()):
                        logger.error("Failed to send response")
                        return
                except Exception:
                    logger.error("Failed to send recovery response", exc_info=True)
            continue


def _get_float_setting(user_settings, key, default):
    """Read a UserSettings value as a float, falling back to `default` on a bad value."""
    value = user_settings.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning(f"Invalid value for setting '{key}': {value!r}, using default {default}")
        return default


def _extract_license_plates(message):
    """Extract (object_id, license_plate) for every object carrying a
    recognized "License Plate Text" attribute.

    Args:
        message: SpeedDetectorMessage with OCR results already applied.

    Returns:
        List of tuples (object_id, license_plate).
    """
    results = []
    for objs in message.objects.values():
        for obj in objs:
            if obj.track_id is None:
                continue
            recognized_text = obj.attributes.get("License Plate Text")
            if not recognized_text or not recognized_text.strip():
                continue
            results.append((obj.track_id, recognized_text.strip()))
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    logger = nxai_postprocessor_sdk.setup_logging()
    processor_config = nxai_postprocessor_sdk.load_config(sys.argv)
    nxai_postprocessor_sdk.apply_config_logging(processor_config, logger)

    logger.info(f"Initializing '{processor_config.get('ProcessorName', PROCESSOR_NAME)}' postprocessor")
    logger.debug(f"Command-line arguments: {sys.argv}")

    ocr_worker_count = int(processor_config.get("ocr_worker_count", DEFAULT_OCR_WORKER_COUNT))
    ocr_output_name = processor_config.get("ocr_output_name", DEFAULT_OCR_OUTPUT_NAME)
    cache_ttl_sec = float(processor_config.get("cache_ttl_sec", DEFAULT_CACHE_TTL_SEC))

    processor = nxai_postprocessor_sdk.ExternalPostprocessor()
    if not processor.initialize_from_args():
        logger.error("Failed to initialize postprocessor from arguments")
        sys.exit(1)
    logger.info(f"Postprocessor '{processor.name}' initialized successfully")

    engine = LogitsOcrEngine(expected_logits_shape=EXPECTED_LOGITS_SHAPE, char_map=CHAR_MAP)
    ocr_pool = OcrWorkerPool(engine, ocr_worker_count)
    ocr_cache = OcrCache(engine, ocr_output_name, ocr_pool, ttl_sec=cache_ttl_sec)
    speed_cache = SpeedMeasurementCache(logger_instance=logger)

    logger.info("Waiting for messages from NXAI runtime...")
    try:
        run_message_loop(processor, ocr_cache, speed_cache)
    except KeyboardInterrupt:
        logger.info("Exited with keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        ocr_pool.stop()
        speed_cache.stop()
        logger.info("Postprocessor shutdown")
