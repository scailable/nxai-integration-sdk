"""
Person Height Estimator Postprocessor (v5 SDK)

Estimates every detected person's height from their bounding box and attaches it
as a "Height" attribute, then fires the 'person.height.exceeded' event
when any person is taller than `max_height_cm`. Calibration is a live
UserSettings control, read fresh from the message every frame,
falling back to the hardcoded defaults when the engine forwards no settings.

The emitted event id is 'person.height.exceeded', exactly as registered in
external_postprocessors.json, with no 'nx.nxai.' prefix.

Behavior change from v4:
- Height is computed from the normalized bbox height: height_m =
  obj.height / frame_fraction_per_meter. The v4 pixel formula
  ((y2 - y1) / pixels_per_meter) is removed — v5 bboxes are frame-relative, so
  no ReceiveInputImage / pixel coordinates are needed. The calibration knob is
  renamed pixels_per_meter -> frame_fraction_per_meter accordingly.
- v4 read calibration from per-frame ExternalProcessorSettings; v5 reads it
  from a live UserSettings control, not from launch-time Config args.
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

PROCESSOR_NAME = "height-estimator"
DEFAULT_FRAME_FRACTION_PER_METER = 0.25
DEFAULT_MAX_HEIGHT_CM = 250  # 2.50 m

# ============================================================================
# Business Logic
# ============================================================================


def handle_message(
    msg: nxai_postprocessor_sdk.ObjectsPostProcessorMessage,
) -> nxai_postprocessor_sdk.ObjectsPostProcessorMessage:
    """Tag every person with an estimated Height and fire on over-limit persons.

    Args:
        msg: Parsed ObjectsPostProcessorMessage from NXAI runtime.

    Returns:
        The same message with a "Height" attribute on each person and, when any
        person exceeds the limit, a 'person.height.exceeded' event
        appended.
    """
    frame_fraction_per_meter = float(
        msg.user_settings.get(
            "frame_fraction_per_meter", DEFAULT_FRAME_FRACTION_PER_METER
        )
    )
    max_height_cm = int(msg.user_settings.get("max_height_cm", DEFAULT_MAX_HEIGHT_CM))
    heights = add_height_attributes(msg, frame_fraction_per_meter)
    max_height_m = max_height_cm / 100.0

    logger.debug(f"Person heights (m): {[round(h, 2) for h in heights]}")

    exceeding = sorted((h for h in heights if h > max_height_m), reverse=True)
    if exceeding:
        description = _format_description(exceeding, len(heights), max_height_m)
        # NOTE: v5 EventMetadata carries only type_id + description; the v4
        # Caption field is dropped — it does not exist in the v5 wire codec.
        event = nxai_postprocessor_sdk.EventMetadata()
        event.type_id = "person.height.exceeded"
        event.description = description
        msg.events.append(event)
        logger.info(f"Height event fired: {description}")

    return msg


# ============================================================================
# Helpers
# ============================================================================


def add_height_attributes(
    msg: nxai_postprocessor_sdk.ObjectsPostProcessorMessage,
    frame_fraction_per_meter: float,
) -> list:
    """Attach a "Height" attribute to every detected person.

    Args:
        msg: Parsed ObjectsPostProcessorMessage.
        frame_fraction_per_meter: Fraction of the frame height spanning 1 metre.

    Returns:
        Estimated height in metres for each detected person.
    """
    heights = []
    for obj in _person_objects(msg):
        height_m = obj.height / frame_fraction_per_meter
        obj.attributes["Height"] = f"{height_m:.2f} m"
        heights.append(height_m)
    return heights


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
# Main Entry Point
# ============================================================================

def _person_objects(msg: nxai_postprocessor_sdk.ObjectsPostProcessorMessage) -> list:
    """Return every object whose class name is "person" (case-insensitive)."""
    persons = []
    for class_name, objects in msg.objects.items():
        if class_name.lower() == "person":
            persons.extend(objects)
    return persons


def _format_description(exceeding: list, total: int, max_height_m: float) -> str:
    """Build a human-readable event description.

    Args:
        exceeding: Heights (m) above the limit, descending.
        total: Number of detected persons.
        max_height_m: The configured limit in metres.

    Returns:
        Description string naming the count, limit, and tallest person.
    """
    return (
        f"{len(exceeding)} of {total} detected person(s) exceed the height limit "
        f"of {max_height_m:.2f} m. Tallest: {exceeding[0]:.2f} m."
    )


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
