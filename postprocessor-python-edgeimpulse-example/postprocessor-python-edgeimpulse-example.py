"""
Edge Impulse Image Uploader Postprocessor (v5 SDK)

Uploads frames to an Edge Impulse project as training samples, either on a timer
(`auto_generator`) or whenever the frame holds a detection the model was unsure
about (`p_value`). Fire-and-forget: the registration JSON sets `"NoResponse":
true`, so nothing is ever sent back.

Behavior-preserving port of the v4 example, with the changes the v5 wire format
forces rather than chooses:

* v4 ran on `Schedule: TENSOR` and hand-unpacked the model's output tensor with
  `struct` to reach every detection's score. A v5 TENSOR-schedule message carries
  no `ImageData`, and this processor needs the score *and* the frame it uploads,
  so it runs on `Schedule: OBJECTS` with `ReceiveInputImage: true` instead. The
  score is read from the decoded objects' `confidence`, which is
  bit-for-bit the same value the tensor row carried.
* The frame arrives at the root `ImageData` key rather than through a separate
  image-header message plus a raw shared-memory read.

One accepted consequence: the score now arrives via the tracker rather than raw
from the model, so a detection on a not-yet-warmed-up first frame may be missed.
"""

import configparser
import io
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
from PIL import Image

import edgeimpulse
import nxai_postprocessor_sdk

# ============================================================================
# Configuration
# ============================================================================

logger = logging.getLogger(__name__)

PROCESSOR_NAME = "edgeimpulse"

script_location = os.path.dirname(sys.argv[0])
CONFIG_FILE = os.path.join(script_location, "..", "etc", "plugin.edgeimpulse.ini")

# Add your own project level Edge Impulse API key to CONFIG_FILE, under
# [edgeimpulse] api_key. The processor refuses to start while it is unset.
DEFAULT_EDGE_IMPULSE_API_KEY = "ei_your_key_here"

# Settings read from CONFIG_FILE at startup, with the v4 defaults.
edge_impulse_api_key = DEFAULT_EDGE_IMPULSE_API_KEY
p_value = 0.4
samples_buffer_flush_size = 20
auto_generator = False
auto_generator_every_seconds = 1

# When the timer path last queued a sample. Armed in __main__ so the first sample
# is due one interval into the run, as the v4 loop's start_time was.
auto_generator_last_sample_at = 0.0

# Frames queued as JPEG bytes, uploaded in one batch.
samples_buffer = []

# Counter behind the unique sample filenames.
samples_counter = 0

# ============================================================================
# Business Logic
# ============================================================================


def handle_message(msg: nxai_postprocessor_sdk.ObjectsPostProcessorMessage) -> None:
    """Queue this frame as a training sample if it is one worth having.

    Which frames those are depends on the mode: on the timer path
    (`auto_generator`) every frame that falls due, otherwise every frame holding
    a detection the model was unsure about.

    Returns nothing: this processor never responds (`NoResponse`).

    Args:
        msg: Parsed ObjectsPostProcessorMessage from NXAI runtime, with
            `image_data` populated (ReceiveInputImage is set in the JSON).
    """
    if not (_sample_is_due() if auto_generator else _has_uncertain_detection(msg)):
        logger.debug("skipping sample")
        return

    logger.debug("uploading sample")
    samples_buffer.append(_jpeg_from_frame(msg.image_data))
    # Logged after the conversion, not before it: this is the line an end-to-end
    # test asserts on, and it has to be false when the frame never
    # became a JPEG.
    logger.info(f"Queued sample: {len(samples_buffer)} pending upload")
    if len(samples_buffer) >= samples_buffer_flush_size:
        send_samples_buffer()


def read_config(config_file: str) -> None:
    """Override the settings above from `config_file`, the processor's ini.

    This processor's settings live in an ini file rather than in the registration
    entry's `Config` block, as they did in v4; a missing or partial file leaves
    the defaults in place.
    """
    global edge_impulse_api_key, p_value, samples_buffer_flush_size
    global auto_generator, auto_generator_every_seconds

    logger.info(f"Reading configuration from: {config_file}")
    configuration = configparser.ConfigParser()
    configuration.read(config_file)
    for section in configuration.sections():
        for key in configuration[section]:
            logger.info(f"config key: {section}.{key} = {configuration[section][key]}")

    # Falls back to whatever the entry's Config LogLevel already set, so an ini
    # without a debug_level does not quietly undo it.
    logger.setLevel(configuration.get("common", "debug_level", fallback=logger.level))
    edge_impulse_api_key = configuration.get(
        "edgeimpulse", "api_key", fallback=DEFAULT_EDGE_IMPULSE_API_KEY)
    auto_generator = configuration.getboolean(
        "edgeimpulse", "auto_generator", fallback=auto_generator)
    auto_generator_every_seconds = configuration.getint(
        "edgeimpulse", "auto_generator_every_seconds", fallback=auto_generator_every_seconds)
    samples_buffer_flush_size = configuration.getint(
        "edgeimpulse", "samples_buffer_flush_size", fallback=samples_buffer_flush_size)
    p_value = configuration.getfloat("edgeimpulse", "p_value", fallback=p_value)


def api_key_is_set() -> bool:
    """Whether the ini replaced the placeholder key; the processor refuses to start
    until it has."""
    return edge_impulse_api_key != DEFAULT_EDGE_IMPULSE_API_KEY


def send_samples_buffer() -> None:
    """Upload every queued frame to Edge Impulse as one batch, and empty the queue.

    Each sample gets a filename unique within the run: two frames can share a
    second, so the counter, not the timestamp, is what separates them.
    """
    global samples_buffer, samples_counter
    if not samples_buffer:
        logger.info(f"No samples to send to Edge Impulse. Total {samples_counter}")
        return

    logger.info(f"Sending {len(samples_buffer)} samples to Edge Impulse...")
    start_at = time.perf_counter()
    samples = []
    for contents in samples_buffer:
        samples_counter += 1
        logger.info(f"Create sample {samples_counter}")
        filename = "{dt}C{c}.jpg".format(
            dt=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), c=samples_counter)
        sample = edgeimpulse.experimental.data.Sample(
            filename=filename,
            data=io.BytesIO(contents),
            metadata={"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        )
        samples.append(sample)

    logger.info("Done creating samples, starting upload...")
    response = edgeimpulse.experimental.data.upload_samples(samples)
    logger.info("Done upload")
    if len(response.fails) != 0:
        logger.info("Could not upload files")

    logger.info(
        "Send {c} samples in {d:0.1f}sec to Edge Impulse. Total {t}".format(
            c=len(samples_buffer), d=time.perf_counter() - start_at, t=samples_counter))
    samples_buffer = []


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
            # Fire-and-forget: perform the side effect and end the loop here.
            # No write_objects_message/send_response call - the AI Manager is
            # configured with "NoResponse": true, and the NoResponse path has
            # no return channel at all.
            handle_message(objects_message)
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            break


# ============================================================================
# Utilities
# ============================================================================


def _sample_is_due() -> bool:
    """Whether `auto_generator_every_seconds` have passed since the last timed
    sample, re-arming the interval when they have."""
    global auto_generator_last_sample_at
    now = time.time()
    if now - auto_generator_last_sample_at < auto_generator_every_seconds:
        return False
    auto_generator_last_sample_at = now
    logger.info(f"Add timed sample every {auto_generator_every_seconds} seconds to upload queue")
    return True


def _has_uncertain_detection(msg: nxai_postprocessor_sdk.ObjectsPostProcessorMessage) -> bool:
    """Whether any detection scored below `p_value` -- the samples worth training
    on are the ones the model was unsure about, so the comparison is `<`."""
    for objects in msg.objects.values():
        for obj in objects:
            if obj.confidence < p_value:
                logger.debug("Parsed value: %.8f", obj.confidence)
                return True
    return False


def _jpeg_from_frame(image_data: nxai_postprocessor_sdk.ImageData) -> bytes:
    """Encode the frame as a JPEG, ready to be handed to Edge Impulse.

    The engine delivers the frame as uint8 RGB, one byte per channel.
    """
    array = np.frombuffer(image_data.bytes, dtype=np.uint8)
    if image_data.layout == nxai_postprocessor_sdk.ImageOrder.CHW:
        array = array.reshape((_CHANNELS, image_data.height, image_data.width)).transpose(1, 2, 0)
    else:
        array = array.reshape((image_data.height, image_data.width, _CHANNELS))
    with Image.fromarray(array) as image:
        with io.BytesIO() as output:
            image.save(output, format="JPEG")
            return output.getvalue()


# RGB, as the v4 example's `Image.frombytes("RGB", ...)` also assumed.
_CHANNELS = 3


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    logger = nxai_postprocessor_sdk.setup_logging()
    processor_config = nxai_postprocessor_sdk.load_config(sys.argv)
    nxai_postprocessor_sdk.apply_config_logging(processor_config, logger)
    logger.info(f"Initializing '{processor_config.get('ProcessorName', PROCESSOR_NAME)}' postprocessor")
    logger.debug(f"Command-line arguments: {sys.argv}")

    # The default sits next to the installed binary, whose directory differs per
    # product; the Config key lets a caller that cannot know that directory --
    # an automated test -- point at a fixed path instead.
    config_file = processor_config.get("config_file", CONFIG_FILE)
    read_config(config_file)
    auto_generator_last_sample_at = time.time()

    if not api_key_is_set():
        logger.error(f"Edge Impulse API key is not set in {config_file}")
        sys.exit(1)
    edgeimpulse.API_KEY = edge_impulse_api_key

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
