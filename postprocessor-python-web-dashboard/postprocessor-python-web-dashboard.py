"""
Web Dashboard Postprocessor (v5 SDK)

Receives inference results from the NX AI Manager, derives per-frame counts and
per-object size/position data, and POSTs them to the companion web app
(web_app.py) for visualization. The objects message is forwarded back to the AI
Manager unmodified — adding the dashboard to a pipeline never changes downstream
results.

The web-app URL is supplied as a launch-time Config arg (--webapp_url <url>),
defaulting to http://localhost:8111.

Behavior change from v4:
- Size/position are derived from the normalized ObjectMetadata bbox
  (size = (width, height), pos = (x + width/2, y + height/2)) and rounded to 4
  decimals; v5 bboxes are frame-relative [0,1] rather than pixel xyxy.
- The frame's pixel Width/Height are no longer sent to the web app: the v5 wire
  codec does not carry them on the objects message (codec-forced drop).
"""

import sys
import json
import logging
import urllib.request

import nxai_postprocessor_sdk

# ============================================================================
# Configuration
# ============================================================================

logger = logging.getLogger(__name__)

PROCESSOR_NAME = "web-dashboard"
DEFAULT_WEBAPP_URL = "http://localhost:8111"
webapp_url = DEFAULT_WEBAPP_URL

# ============================================================================
# Business Logic
# ============================================================================


def handle_message(
    msg: nxai_postprocessor_sdk.ObjectsPostProcessorMessage,
) -> nxai_postprocessor_sdk.ObjectsPostProcessorMessage:
    """Forward per-frame detection data to the web app, return msg unmodified.

    Args:
        msg: Parsed ObjectsPostProcessorMessage from the NXAI runtime.

    Returns:
        The same message, untouched — the dashboard is a read-only observer.
    """
    counts = count_objects(msg)
    sizes, positions = extract_bbox_data(msg)
    payload = {
        "ts": msg.timestamp,
        "counts": counts,
        "sizes": sizes,
        "positions": positions,
    }
    post_to_webapp(webapp_url, payload)
    logger.debug(f"Forwarded frame ts={msg.timestamp} counts={counts}")
    return msg


def run_message_loop(processor: nxai_postprocessor_sdk.ExternalPostprocessor) -> None:
    """Serve messages until the engine sends a graceful EXIT signal.

    `wait_for_message()` returns None both on a plain timeout and on an IPC EXIT
    signal; only `processor.exit_requested` tells them apart. A loop that just
    `continue`s on None regardless of that flag never returns, and the AI Manager
    kills the process after its grace period instead of it exiting cleanly.

    The dashboard's HTTP server and periodic flush belong to the companion
    `web_app.py`, not to this process; the only thing here that could hold the exit
    up is the POST, whose wait `post_to_webapp` bounds.

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
    """Return the number of detected objects per class via len(msg.objects[cls])."""
    return {class_name: len(objects) for class_name, objects in msg.objects.items()}


def extract_bbox_data(msg: nxai_postprocessor_sdk.ObjectsPostProcessorMessage):
    """Return flat lists of size and position dicts, each carrying a class label 'c'.

    Size is the normalized bbox extent (width, height); position is its centre
    (x + width/2, y + height/2). Coordinates are normalized [0,1], rounded to 4
    decimals.
    """
    sizes, positions = [], []
    for class_name, objects in msg.objects.items():
        for obj in objects:
            sizes.append(
                {"x": round(obj.width, 4), "y": round(obj.height, 4), "c": class_name}
            )
            positions.append(
                {
                    "x": round(obj.x + obj.width / 2, 4),
                    "y": round(obj.y + obj.height / 2, 4),
                    "c": class_name,
                }
            )
    return sizes, positions


def post_to_webapp(url: str, payload: dict) -> bool:
    """POST one frame of detection data to the web app (fire-and-forget)."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url + "/api/ingest",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception as e:
        logger.warning(f"Could not reach web app at {url}: {e}")
        return False


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    logger = nxai_postprocessor_sdk.setup_logging()
    processor_config = nxai_postprocessor_sdk.load_config(sys.argv)
    nxai_postprocessor_sdk.apply_config_logging(processor_config, logger)
    logger.info(f"Initializing '{processor_config.get('ProcessorName', PROCESSOR_NAME)}' postprocessor")
    logger.debug(f"Command-line arguments: {sys.argv}")

    if "webapp_url" in processor_config:
        webapp_url = processor_config["webapp_url"]
    logger.info(f"Web app URL: {webapp_url}")

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
