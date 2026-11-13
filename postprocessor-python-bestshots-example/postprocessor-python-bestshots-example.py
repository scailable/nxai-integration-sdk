"""
BestShots Postprocessor (v5 SDK)

Tags an already-tracked "car" object (a base-detector class, not one this
processor injects) with a BestShots entry each frame, gated by the
`emit_best_shots` launch-time Config flag ("true"/"false", default "false").

Fixture for an end-to-end check that a BestShots entry emitted by a
postprocessor reaches the VMS: one processor whose Config toggles the
positive/negative scenario, mirroring the
`OBJECT_COUNTER_THRESHOLD`/`min_object_count` Config pattern rather than two
separate binaries. The engine correlates a BestShots entry with the object
by class and track id.

Only an already-tracked car is eligible (`track_id` set, i.e. the object was
seen on a prior frame) -- not a car detected for the first time this frame
(`new_track_id` only, no `track_id` yet), because BestShots names an
existing tracker-assigned UUID, it never mints a new one.
"""

import sys
import logging
import nxai_postprocessor_sdk

# ============================================================================
# Configuration
# ============================================================================

logger = logging.getLogger(__name__)

PROCESSOR_NAME = "bestshots-example"
TARGET_CLASS = "car"

# Synthetic sentinel best-shot rectangle a real detector could never plausibly
# produce -- must match check_scenario.py's _SENTINEL_BBOX exactly. Emitting the
# tracked object's own per-frame position here would be indistinguishable from
# the engine's own auto-populated best_shot() (live-verified 2026-07-15: every
# track gets a real, non-empty boundingBox regardless of whether a BestShots
# entry is ever sent), so the e2e proof needs a value only this processor emits.
_SENTINEL_BBOX = (0.125, 0.0625, 0.03125, 0.03125)

emit_best_shots = False

# ============================================================================
# Business Logic
# ============================================================================


def handle_message(
    msg: nxai_postprocessor_sdk.ObjectsPostProcessorMessage,
) -> nxai_postprocessor_sdk.ObjectsPostProcessorMessage:
    """Tag the first already-tracked TARGET_CLASS object as a best shot.

    No-op unless `emit_best_shots` is true and an already-tracked TARGET_CLASS
    object is present in this frame.

    Args:
        msg: Parsed ObjectsPostProcessorMessage from NXAI runtime.

    Returns:
        The same message, possibly with a best shot appended.
    """
    if not emit_best_shots:
        return msg

    target = _first_tracked(msg.objects.get(TARGET_CLASS, []))
    if target is None:
        logger.debug(f"No already-tracked '{TARGET_CLASS}' object this frame")
        return msg

    best_shot = nxai_postprocessor_sdk.BestShotMetadata()
    best_shot.track_id = target.track_id
    best_shot.x, best_shot.y, best_shot.width, best_shot.height = _SENTINEL_BBOX
    msg.best_shots.append(best_shot)
    logger.info(f"Tagged track {best_shot.track_id} as best shot")

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


def _first_tracked(objects) -> nxai_postprocessor_sdk.ObjectMetadata:
    """Return the first already-tracked object (track_id set), or None.

    An object first detected this frame carries only new_track_id -- BestShots
    names an existing tracker-assigned UUID, so such an object is not eligible.
    """
    for obj in objects:
        if obj.track_id is not None:
            return obj
    return None


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    logger = nxai_postprocessor_sdk.setup_logging()
    processor_config = nxai_postprocessor_sdk.load_config(sys.argv)
    nxai_postprocessor_sdk.apply_config_logging(processor_config, logger)
    logger.info(f"Initializing '{processor_config.get('ProcessorName', PROCESSOR_NAME)}' postprocessor")
    logger.debug(f"Command-line arguments: {sys.argv}")

    if "emit_best_shots" in processor_config:
        emit_best_shots = processor_config["emit_best_shots"].strip().lower() == "true"
    logger.info(f"emit_best_shots: {emit_best_shots}")

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
