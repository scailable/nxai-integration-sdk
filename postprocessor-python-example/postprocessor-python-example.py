"""
Example NXAI postprocessor using the new SDK.

This postprocessor receives detected objects and events from the NXAI runtime,
adds a test object to demonstrate processing, and sends the modified results back.

Configuration:
  - Logs: written to stdout/stderr; the AI Manager writes them to the entry's LogFilepath
  - Config: the Config keys of the registration entry, passed as launch arguments
"""

import sys
import logging
import uuid
import nxai_postprocessor_sdk

# ============================================================================
# Configuration
# ============================================================================

# Module-level logger
logger = logging.getLogger(__name__)

PROCESSOR_NAME = "python-example"

# Config received from AI Manager
global processor_config

# Class of the injected object. Must match the sole Objects[].ID declared in this
# example's external_postprocessors.json, and must be unique across every processor
# registered on the server: the AI Manager merges all registered processors' Objects
# into every device's manifest without deduplicating, so two processors sharing an ID
# make the manifest report duplicate object types.
INJECTED_CLASS = "python_example"

# UUID of the injected object's track, per device: assigned on the first
# handle_message call for a device and reused on every subsequent call for it, so
# the track persists across frames. Keyed by device because a track belongs to
# the device that minted it -- the plugin resolves an incoming ID against that
# device's own track table, so reusing one camera's ID on another is rejected
# with "Could not find object ID" and the object is dropped.
global test_track_ids
test_track_ids = {}

# ============================================================================
# Business Logic
# ============================================================================


def handle_message(
    msg: nxai_postprocessor_sdk.ObjectsPostProcessorMessage,
) -> nxai_postprocessor_sdk.ObjectsPostProcessorMessage:
    """Process a received message and return modified message.

    Args:
        msg: Parsed ObjectsPostProcessorMessage from NXAI runtime

    Returns:
        Modified ObjectsPostProcessorMessage to send back to runtime
    """

    logger.info(
        f"Processing: device={msg.device_id}, "
        f"{len(msg.objects)} object classes, {len(msg.events)} events"
    )
    logger.debug(f"Device: {msg.device_name}, Timestamp: {msg.timestamp}")

    # Example: add a test object under this example's own class
    if INJECTED_CLASS not in msg.objects:
        msg.objects[INJECTED_CLASS] = []

    # Create a test object, continuing the same track across frames so the
    # object persists in the analytics archive instead of flickering as a new
    # single-frame track every time.
    test_obj = nxai_postprocessor_sdk.ObjectMetadata()
    test_track_id = test_track_ids.get(msg.device_id)
    if test_track_id is None:
        test_track_id = str(uuid.uuid4())
        test_track_ids[msg.device_id] = test_track_id
        test_obj.new_track_id = test_track_id  # create the track with our chosen UUID
    else:
        test_obj.track_id = test_track_id  # continue this device's existing track
    test_obj.x = 0.1
    test_obj.y = 0.1
    test_obj.width = 0.3
    test_obj.height = 0.3
    test_obj.confidence = 0.95
    test_obj.attributes["source"] = "postprocessor-example"

    msg.objects[INJECTED_CLASS].append(test_obj)
    logger.info(f"Added test object ")

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
            # Wait for message from server (timeout in milliseconds)
            message_bytes = processor.wait_for_message(timeout_ms=5000)

            if message_bytes is None:
                if processor.exit_requested:
                    logger.info("Received graceful exit signal, shutting down")
                    return
                logger.debug("Timeout waiting for message")
                continue

            logger.debug(f"Received message ({len(message_bytes)} bytes)")

            # Parse msgpack message into ObjectsPostProcessorMessage
            objects_message = processor.parse_objects_message(message_bytes)
            if objects_message is None:
                logger.error("Failed to parse message")
                continue

            # Process message using business logic
            processed_msg = handle_message(objects_message)

            # Serialize message back to bytes
            response_bytes = processor.write_objects_message(processed_msg)
            if response_bytes is None:
                logger.error("Failed to serialize message")
                continue

            # Send response back to server
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


if __name__ == "__main__":
    # Set up logging using SDK utility
    logger = nxai_postprocessor_sdk.setup_logging()

    # Load configuration using SDK utility
    processor_config = nxai_postprocessor_sdk.load_config(sys.argv)

    # Apply logging configuration if available
    nxai_postprocessor_sdk.apply_config_logging(processor_config, logger)

    # Log startup information
    logger.info(f"Initializing '{processor_config.get('ProcessorName', PROCESSOR_NAME)}' postprocessor")
    logger.debug(f"Command-line arguments: {sys.argv}")

    # Initialize ExternalPostprocessor from command-line arguments
    processor = nxai_postprocessor_sdk.ExternalPostprocessor()
    if not processor.initialize_from_args():
        logger.error("Failed to initialize postprocessor from arguments")
        sys.exit(1)

    logger.info(f"Postprocessor '{processor.name}' initialized successfully")
    logger.info("Waiting for messages from NXAI runtime...")

    # Main message processing loop
    try:
        run_message_loop(processor)
    except KeyboardInterrupt:
        logger.info("Exited with keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("Postprocessor shutdown")
