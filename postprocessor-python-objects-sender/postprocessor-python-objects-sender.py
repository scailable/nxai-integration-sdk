"""
Objects Sender Postprocessor (v5 SDK)

Geo-references detections. The operator marks up to 10 reference points on the
camera view (BoxFigure) and gives each a latitude and a longitude. With at least
4 points, the processor computes a homography per device, which maps the center
of each object's bbox to (lat, lon). It adds a "Coordinates" attribute to every
object and POSTs the objects as GeoJSON features to an HTTP-to-MQTT bridge at
<server_address>/send. A failed request is logged, and the message is returned.

Behavior change from v4:
- The homography is computed in normalized frame coordinates. v4 scaled the
  figure points and the bbox centers to pixels with the same frame Width/Height;
  the v5 OBJECTS message carries no frame size, and the scaling is absorbed into
  the homography, so (lat, lon) is the same.
- The homography is recomputed only when the reference points of the device
  change. v4 reset its cache on every frame and recomputed it every time.
- A reference point is read with its own latitude and longitude. v4 matched them
  by the Repeater index, and a gap in the indexes (point 0 empty, point 1 set)
  raised IndexError and stopped the processor.
"""

import json
import logging
import sys

import cv2
import numpy as np
import requests

import nxai_postprocessor_sdk

# ============================================================================
# Configuration
# ============================================================================

logger = logging.getLogger(__name__)

PROCESSOR_NAME = "objects-sender"
MAX_POINTS = 10
MIN_POINTS = 4
# The Nx spin boxes keep three decimals, so the latitude and longitude settings
# carry degrees times 1000.
MANTISSA_COEFFICIENT = 1000
DEFAULT_LATITUDE = 33878.754
DEFAULT_LONGITUDE = -84450.569
DEFAULT_SERVER_ADDRESS = "http://127.0.0.1:5000"
DEFAULT_TOPIC_PATH = "topic1"
REQUEST_TIMEOUT_S = 10

# device_id -> homography of that device, or None with fewer than MIN_POINTS points
_homographies = {}
# device_id -> the reference points _homographies was computed from
_known_points_cache = {}

# ============================================================================
# Business Logic
# ============================================================================


def handle_message(
    msg: nxai_postprocessor_sdk.ObjectsPostProcessorMessage,
) -> nxai_postprocessor_sdk.ObjectsPostProcessorMessage:
    """Add a "Coordinates" attribute to every object and send the objects to MQTT.

    Args:
        msg: Parsed ObjectsPostProcessorMessage from NXAI runtime.

    Returns:
        The same message. Without reference points it is unchanged.
    """
    device_id = msg.device_id
    known_points = read_known_points(msg.user_settings)
    if not known_points:
        return msg

    if known_points != _known_points_cache.get(device_id):
        _known_points_cache[device_id] = known_points
        _homographies[device_id] = (
            compute_homography(known_points) if len(known_points) >= MIN_POINTS else None
        )

    homography = _homographies[device_id]
    logger.debug(homography)
    if homography is None:
        return msg

    timestamp = float(msg.timestamp)
    message = []
    for class_name, objects in msg.objects.items():
        for obj in objects:
            bbox_center = (obj.x + obj.width / 2, obj.y + obj.height / 2)
            lat, lon = apply_homography(homography, bbox_center)
            obj.attributes["Coordinates"] = f"{lat}, {lon}"
            message.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [str(lat), str(lon)]},
                    "properties": {
                        "object_type": str(class_name),
                        "object_id": str(obj.track_id),
                        "timestamp": str(timestamp),
                    },
                }
            )

    if message:
        mqtt_address = str(msg.user_settings.get("server_address", DEFAULT_SERVER_ADDRESS))
        topic_path = str(msg.user_settings.get("topic_path", DEFAULT_TOPIC_PATH))
        send_to_mqtt(mqtt_address, topic_path, message)

    return msg


def read_known_points(user_settings: dict) -> list:
    """Read the configured reference points from the UserSettings map.

    Args:
        user_settings: The flat {name: value} map of msg.user_settings.

    Returns:
        One {"pixel": (x, y), "lat_lon": (lat, lon)} entry per point that has a
        figure. "pixel" is the normalized center of the figure's box.
    """
    known_points = []
    for i in range(MAX_POINTS):
        key_string = f"point{i}.figure"
        if key_string not in user_settings:
            continue

        figure = json.loads(user_settings[key_string])
        logger.debug(figure)
        if figure.get("figure") is None:
            continue

        points = figure["figure"]["points"]
        box_center = (
            (points[1][0] - points[0][0]) / 2 + points[0][0],
            (points[1][1] - points[0][1]) / 2 + points[0][1],
        )
        latitude = float(user_settings.get(f"point{i}.latitude", DEFAULT_LATITUDE))
        longitude = float(user_settings.get(f"point{i}.longitude", DEFAULT_LONGITUDE))
        known_points.append(
            {
                "pixel": box_center,
                "lat_lon": (latitude / MANTISSA_COEFFICIENT, longitude / MANTISSA_COEFFICIENT),
            }
        )
    return known_points


def compute_homography(known_points: list):
    """Compute the homography from the reference points with OpenCV (RANSAC).

    Returns:
        The 3x3 homography, or None if OpenCV cannot fit one.
    """
    logger.debug("computing homography")

    pixel_points = np.array([point["pixel"] for point in known_points], dtype=np.float32)
    real_world_points = np.array([point["lat_lon"] for point in known_points], dtype=np.float32)

    homography, _ = cv2.findHomography(pixel_points, real_world_points, method=cv2.RANSAC)
    return homography


def apply_homography(homography, pixel_coord: tuple) -> tuple:
    """Apply a homography to a normalized frame coordinate."""
    logger.debug("applying homography")
    px, py = pixel_coord
    transformed = np.dot(homography, np.array([px, py, 1]))
    x, y = transformed[:2] / transformed[2]  # Normalize by Z
    return (x, y)


def send_to_mqtt(mqtt_address: str, topic_path: str, message: list) -> None:
    """POST the GeoJSON features to the HTTP-to-MQTT bridge. Log a failure.

    Args:
        mqtt_address: Base URL of the bridge; the request goes to <mqtt_address>/send.
        topic_path: MQTT topic the bridge publishes to.
        message: GeoJSON Feature list, one entry per object.
    """
    url = mqtt_address + "/send"
    body = {"topic": topic_path, "QoS": 1, "message": message}
    try:
        response = requests.request("POST", url, json=body, timeout=REQUEST_TIMEOUT_S)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to the server.")
    except requests.exceptions.Timeout:
        logger.error("The request timed out.")
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error occurred: {e.response.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"An unexpected request error occurred: {e}")


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
