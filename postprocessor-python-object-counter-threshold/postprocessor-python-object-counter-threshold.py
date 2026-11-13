"""
Object Counter Threshold Postprocessor (v5 SDK)

Counts the number of detected objects per class each frame and fires the
'object.counter.threshold' event when the total count meets or exceeds
`min_object_count`. The threshold is a live UserSettings control,
read fresh from the message every frame, falling back to
DEFAULT_MIN_OBJECT_COUNT when the engine forwards no settings.

The emitted event id is 'object.counter.threshold', exactly as registered in
external_postprocessors.json, with no 'nx.nxai.' prefix.

Behavior change from v4:
- v4 read the threshold from per-frame ExternalProcessorSettings; v5 reads it
  from a live UserSettings control, not from a launch-time Config arg.
- v4 emitted a Caption field on the event; v5 EventMetadata carries only
  type_id + description (Caption dropped: the v5 wire codec has no Caption field).
"""

import sys
import logging
import nxai_postprocessor_sdk

# ============================================================================
# Configuration
# ============================================================================

logger = logging.getLogger(__name__)

PROCESSOR_NAME = "object-counter-threshold"
DEFAULT_MIN_OBJECT_COUNT = 5

# ============================================================================
# Business Logic
# ============================================================================


def handle_message(
    msg: nxai_postprocessor_sdk.ObjectsPostProcessorMessage,
) -> nxai_postprocessor_sdk.ObjectsPostProcessorMessage:
    """Fire 'object.counter.threshold' when total object count >= min_object_count.

    Args:
        msg: Parsed ObjectsPostProcessorMessage from NXAI runtime.

    Returns:
        The same message, possibly with an event appended.
    """
    min_object_count = int(
        msg.user_settings.get("min_object_count", DEFAULT_MIN_OBJECT_COUNT)
    )
    counts = count_objects(msg)
    total = sum(counts.values())

    logger.debug(f"Object counts: {counts}, total={total}, threshold={min_object_count}")

    if total >= min_object_count:
        description = _format_description(counts, total)
        # NOTE: v5 EventMetadata carries only type_id + description; the v4
        # Caption field is dropped — it does not exist in the v5 wire codec.
        event = nxai_postprocessor_sdk.EventMetadata()
        event.type_id = "object.counter.threshold"
        event.description = description
        msg.events.append(event)
        logger.info(f"Threshold met ({total} >= {min_object_count}): event appended")
    else:
        logger.debug(f"Threshold not met ({total} < {min_object_count}): no event")

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


def count_objects(msg: nxai_postprocessor_sdk.ObjectsPostProcessorMessage) -> dict:
    """Return {class_name: count} for every non-empty class in msg.objects.

    Args:
        msg: Parsed ObjectsPostProcessorMessage.

    Returns:
        Dict mapping class name to object count; empty classes omitted.
    """
    # Skip classes with no detections so they never reach the description.
    return {cls: len(objs) for cls, objs in msg.objects.items() if objs}


def _format_description(counts: dict, total: int) -> str:
    """Build a human-readable event description.

    Args:
        counts: {class_name: count} from count_objects().
        total: Pre-computed sum of all counts.

    Returns:
        Description string, e.g. "Detected 2 car, 3 person. Total: 5 objects."
    """
    parts = [f"{n} {cls}" for cls, n in sorted(counts.items())]
    return f"Detected {', '.join(parts)}. Total: {total} object{'s' if total != 1 else ''}."


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
