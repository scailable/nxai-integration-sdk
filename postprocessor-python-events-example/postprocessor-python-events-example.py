"""
Events Postprocessor (v5 SDK)

Emits one 'ex.example.event' event every frame, with a description
listing the number of detected objects per class.

Behavior change from v4:
- v4 emitted a Caption field on the event; v5 EventMetadata carries only
  type_id + description (Caption dropped: the v5 wire codec has no Caption
  field).
- The per-class count came from `len(coordinates) / 4` on v4's flat xyxy
  arrays; v5's msg.objects already holds one entry per detected object, so
  the count is `len(class_objects)` directly. Same displayed number.

Preserved from v4 (not a bug fix): the description-building loop is order
dependent on msg.objects' class iteration order. A class with zero
detections that appears before the first class with a nonzero count has its
partial "no objects" text fully discarded rather than appended to; classes
encountered after the first nonzero-count class are appended normally
regardless of their own count.
"""

import sys
import logging
import nxai_postprocessor_sdk

# ============================================================================
# Configuration
# ============================================================================

logger = logging.getLogger(__name__)

PROCESSOR_NAME = "events-example"

# ============================================================================
# Business Logic
# ============================================================================


def handle_message(
    msg: nxai_postprocessor_sdk.ObjectsPostProcessorMessage,
) -> nxai_postprocessor_sdk.ObjectsPostProcessorMessage:
    """Append one 'ex.example.event' event describing detected objects.

    Args:
        msg: Parsed ObjectsPostProcessorMessage from NXAI runtime.

    Returns:
        The same message, with one event appended.
    """
    description = _format_description(msg.objects)
    # NOTE: v5 EventMetadata carries only type_id + description; the v4
    # Caption field is dropped — it does not exist in the v5 wire codec.
    event = nxai_postprocessor_sdk.EventMetadata()
    event.type_id = "ex.example.event"
    event.description = description
    msg.events.append(event)
    logger.info("Added test event to output")

    return msg


def run_message_loop(processor: nxai_postprocessor_sdk.ExternalPostprocessor) -> None:
    """Serve messages until the engine sends a graceful EXIT signal.

    `wait_for_message()` returns None both on a plain timeout and on an IPC EXIT
    signal; only `processor.exit_requested` tells them apart. A loop that just
    `continue`s on None regardless of that flag never returns, and the AI Manager
    kills the process after its grace period instead of it exiting cleanly.

    Args:
        processor: An initialized ExternalPostprocessor.
    """
    while True:
        logger.debug("Waiting for input message")
        try:
            message_bytes = processor.wait_for_message(timeout_ms=5000)
            if message_bytes is None:
                if processor.exit_requested:
                    logger.info("Received graceful exit signal, shutting down")
                    return
                logger.debug("Timeout waiting for message")
                continue
            logger.debug(f"Received message ({len(message_bytes)} bytes)")
            objects_message = processor.parse_objects_message(message_bytes)
            if objects_message is None:
                logger.error("Failed to parse message")
                continue
            processed_msg = handle_message(objects_message)
            response_bytes = processor.write_objects_message(processed_msg)
            if response_bytes is None:
                logger.error("Failed to serialize message")
                continue
            if not processor.send_response(response_bytes):
                logger.error("Failed to send response")
                break
            logger.debug("Response sent successfully")
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            break


# ============================================================================
# Helpers
# ============================================================================


def _format_description(objects: dict) -> str:
    """Build the per-class object-count description, v4-order-dependent quirk intact.

    Args:
        objects: msg.objects, {class_name: [ObjectMetadata, ...]}.

    Returns:
        Description string, e.g. "\\nThere are 2 person's in the frame".
    """
    description = "\nThere are no objects in the frame."
    objects_detected = False
    for class_name, class_objects in objects.items():
        if not objects_detected and len(class_objects) > 0:
            description = "\nThere are"
            objects_detected = True
        description += " " + str(len(class_objects)) + " " + class_name + "'s in the frame"
    return description


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    logger = nxai_postprocessor_sdk.setup_logging()
    processor_config = nxai_postprocessor_sdk.load_config(sys.argv)
    nxai_postprocessor_sdk.apply_config_logging(processor_config, logger)
    logger.info(f"Initializing '{processor_config.get('ProcessorName', PROCESSOR_NAME)}' postprocessor")
    logger.debug(f"Command-line arguments: {sys.argv}")

    processor = nxai_postprocessor_sdk.ExternalPostprocessor()
    if not processor.initialize_from_args():
        logger.error("Failed to initialize postprocessor from arguments")
        sys.exit(1)
    logger.info(f"Postprocessor '{processor.name}' initialized successfully")
    logger.info("Waiting for messages from NXAI runtime...")
    try:
        run_message_loop(processor)
    except KeyboardInterrupt:
        logger.info("Exited with keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("Postprocessor shutdown")
