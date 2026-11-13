#!/usr/bin/env python3
"""
Postprocessor Python ANPR Example (v5 SDK)

Assigned to the OCR model chained after the plate detector in Feature
Extraction mode. Each message carries one plate crop's OCR result (NxM float32
logits, delivered as a raw Tensors block on an OBJECTS-schedule message) and
the plate's track ID as OriginalObjectID. The processor
decodes the logits and answers with an attribute-only update of that original
plate object: "License Plate Text" / "Confidence" are set to the best reading
seen for the plate so far, or removed when there is none. Emits a deduplicated
'anpr.plate_recognized' event once a plate's text clears the configured
confidence threshold.

Behavior change from the earlier v5 version:
- The processor is assigned to the OCR model only. It used to be assigned to
  both models, caching OCR results from the chained stage and attaching them
  to the detector's message for the same object on a later frame. The engine
  now lets a chained-stage processor edit the original object by ID, so the
  plate text lands on the same frame and no detector-stage hop is needed.
- OCR decoding is synchronous (the result has to be in this message's
  response), so the OCR worker pool and its `ocr_worker_count` Config value
  are gone.

Behavior change from v4:
- Confidence threshold (`min_confidence`) is now a live UserSettings control,
  read fresh from each message with a hardcoded default
  fallback, instead of an INI file value overridden by a
  `externalprocessor.`-prefixed ExternalProcessorSettings key.
- OCR output tensor name, cache TTL and the plate class name are launch-time
  Config values (`--key value` in external_postprocessors.json), replacing the
  removed `plugin.anpr.ini` file.
- The per-socket-path singleton lock is removed: v5 launches exactly one
  process per registration entry over its own dedicated IPC channels, so there
  is no shared socket path to race on.
- The emitted event drops the v4 Caption field (v5 EventMetadata carries only
  type_id + description) and carries the id 'anpr.plate_recognized' exactly as
  registered in external_postprocessors.json, with no 'nx.nxai.' prefix.
"""

import os
import sys
import time
import logging

script_location = os.path.dirname(os.path.abspath(__file__))
if os.path.join(script_location, "..") not in sys.path:
    sys.path.insert(0, os.path.join(script_location, ".."))

import nxai_postprocessor_sdk
from message_processing_utils import create_anpr_message_from_bytes
from message_processing_utils.anpr import CctOcrMessage, EventDeduplicationCache
from message_processing_utils.anpr.attributes import LICENSE_PLATE_CONFIDENCE, LICENSE_PLATE_TEXT
from message_processing_utils.general.ocr import LogitsOcrEngine, OcrCache

logger = logging.getLogger(__name__)

PROCESSOR_NAME = "anpr-example"
EVENT_ID_PLATE_RECOGNIZED = "anpr.plate_recognized"
PLATE_ATTRIBUTES = [LICENSE_PLATE_TEXT, LICENSE_PLATE_CONFIDENCE]

DEFAULT_MIN_CONFIDENCE = 0.95
DEFAULT_OCR_OUTPUT_NAME = "Identity:0"
DEFAULT_CACHE_TTL_SEC = 300.0
# Class of the detector model's plate objects (License_Plate_Detector.onnx
# outputs "bboxes-format:xyxysc;0:LP"). The engine finds the original object by
# its ID regardless, but the class it's returned under is what the chained
# stage's visualisation settings are checked against.
DEFAULT_PLATE_CLASS_NAME = "LP"
EXPECTED_LOGITS_SHAPE = (9, 37)
CHAR_MAP = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "

# ============================================================================
# Business Logic
# ============================================================================


def handle_message(message, ocr_cache, event_cache, min_confidence, plate_class_name=DEFAULT_PLATE_CLASS_NAME):
    """Decode this message's OCR result onto its original plate object.

    Args:
        message: AnprDetectorMessage or CctOcrMessage from create_anpr_message_from_bytes.
        ocr_cache: OcrCache holding the best reading per plate, keyed by original_object_id.
        event_cache: EventDeduplicationCache shared across messages.
        min_confidence: Effective confidence threshold for emitting an event.
        plate_class_name: Class the original plate object is returned under.

    Returns:
        The same message. On a chained-stage message the original plate object
        gets the best reading so far as License Plate Text/Confidence, or has
        both attributes removed when there is no reading, so a plate never
        keeps text from an earlier or stale result.
    """
    object_id = message.original_object_id
    if not object_id:
        # Not a message from the chained OCR model: this processor is no
        # longer assigned to the detector, so there is nothing to add.
        logger.debug("Message without OriginalObjectID, returning it unchanged")
        return message

    reading = _decode_reading(message, ocr_cache)
    if reading is not None:
        ocr_cache.cache_ocr_result(message, *reading)

    best = ocr_cache.get_cached_result(object_id)
    if best is None:
        message.update_original_object(plate_class_name, remove_attributes=PLATE_ATTRIBUTES)
        logger.debug("No plate read for object %s, removing plate attributes", object_id)
        return message

    text, confidence = best
    message.update_original_object(
        plate_class_name,
        attributes={
            LICENSE_PLATE_TEXT: text,
            LICENSE_PLATE_CONFIDENCE: f"{float(confidence):.4f}",
        },
    )
    _emit_plate_event(message, object_id, text, confidence, event_cache, min_confidence)
    return message


def run_message_loop(processor, ocr_cache, event_cache, plate_class_name=DEFAULT_PLATE_CLASS_NAME):
    """Serve messages until the engine stops sending or the response channel fails.

    A failure while handling one message must not end the loop: the engine keeps
    sending frames, and a processor that exits is dropped from the pipeline.

    Args:
        processor: An initialized ExternalPostprocessor.
        ocr_cache: OcrCache shared across messages.
        event_cache: EventDeduplicationCache shared across messages.
        plate_class_name: Class the original plate object is returned under.
    """
    while True:
        logger.debug("Waiting for input message")
        message = None
        try:
            now = time.monotonic()
            ocr_cache.cleanup(now)
            event_cache.cleanup(now)

            message_bytes = processor.wait_for_message(timeout_ms=5000)
            if message_bytes is None:
                if processor.exit_requested:
                    logger.info("Received graceful exit signal, shutting down")
                    return
                logger.debug("Timeout waiting for message")
                continue

            logger.debug(f"Received message ({len(message_bytes)} bytes)")

            message = create_anpr_message_from_bytes(message_bytes)
            if message is None:
                logger.warning("Could not parse message, skipping")
                continue

            min_confidence = _get_float_setting(
                message.user_settings, "min_confidence", DEFAULT_MIN_CONFIDENCE
            )
            message = handle_message(message, ocr_cache, event_cache, min_confidence, plate_class_name)

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


def _decode_reading(message, ocr_cache):
    """Decode this message's OCR logits, or None if they hold no plate text.

    A chained message without a usable Tensors block (ReceiveBinaryData off,
    wrong tensor name or shape) is parsed as a detector message rather than a
    CctOcrMessage, and also yields None.
    """
    if not isinstance(message, CctOcrMessage):
        logger.warning(
            "Chained message for object %s has no OCR tensors; is ReceiveBinaryData enabled?",
            message.original_object_id,
        )
        return None
    result = message.decode_ocr_logits(ocr_cache.engine, ocr_cache.output_name)
    if not result:
        return None
    text, confidence = result
    if not text:
        # Every position decoded to the blank character.
        return None
    return text, confidence


def _get_float_setting(user_settings, key, default):
    """Read a UserSettings value as a float, falling back to `default` on a bad value."""
    value = user_settings.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning(f"Invalid value for setting '{key}': {value!r}, using default {default}")
        return default


def _emit_plate_event(message, object_id, text, confidence, event_cache, min_confidence):
    """Emit a deduplicated plate_recognized event if the reading clears the threshold."""
    if confidence < min_confidence:
        return
    if not event_cache.should_generate_event(object_id, text):
        return
    message.add_event(EVENT_ID_PLATE_RECOGNIZED, text)
    event_cache.mark_event_generated(object_id, text, time.monotonic())
    logger.debug(
        "Emitted plate_recognized event for object %s: %s (conf=%.4f)",
        object_id, text, confidence,
    )


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    logger = nxai_postprocessor_sdk.setup_logging()
    processor_config = nxai_postprocessor_sdk.load_config(sys.argv)
    nxai_postprocessor_sdk.apply_config_logging(processor_config, logger)

    logger.info(f"Initializing '{processor_config.get('ProcessorName', PROCESSOR_NAME)}' postprocessor")
    logger.debug(f"Command-line arguments: {sys.argv}")

    ocr_output_name = processor_config.get("ocr_output_name", DEFAULT_OCR_OUTPUT_NAME)
    cache_ttl_sec = float(processor_config.get("cache_ttl_sec", DEFAULT_CACHE_TTL_SEC))
    plate_class_name = processor_config.get("plate_class_name", DEFAULT_PLATE_CLASS_NAME)

    processor = nxai_postprocessor_sdk.ExternalPostprocessor()
    if not processor.initialize_from_args():
        logger.error("Failed to initialize postprocessor from arguments")
        sys.exit(1)
    logger.info(f"Postprocessor '{processor.name}' initialized successfully")

    engine = LogitsOcrEngine(expected_logits_shape=EXPECTED_LOGITS_SHAPE, char_map=CHAR_MAP)
    ocr_cache = OcrCache(engine, ocr_output_name, ttl_sec=cache_ttl_sec)
    event_cache = EventDeduplicationCache(ttl_sec=cache_ttl_sec)

    logger.info("Waiting for messages from NXAI runtime...")
    try:
        run_message_loop(processor, ocr_cache, event_cache, plate_class_name)
    except KeyboardInterrupt:
        logger.info("Exited with keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("Postprocessor shutdown")
