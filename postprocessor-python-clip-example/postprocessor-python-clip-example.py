#!/usr/bin/env python3
"""
Postprocessor Python CLIP Example (v5 SDK)

Pairs with `preprocessor-python-clip-example`: that preprocessor tokenizes up to
five operator-configured prompts into a chained CLIP model's "text" tensor; this
postprocessor reads the CLIP model's "scores" output tensor (raw `Tensors`, opt-in
via `ReceiveBinaryData: true`) and picks the winning prompt with an `argmax`,
mirroring v4's `argmax` over the engine-computed `Scores` map.

The CLIP-scoring message carries `OriginalObjectID` but an empty `Objects` map
(CLIP is not a detector), so the winning prompt is cached (module-level
`objects_attributes`, FIFO-evicted past 100 entries, ported from v4 unchanged) and
applied as the `clip_prompt` attribute to the same object on a later OBJECTS
message from the detector node.

Behavior change from v4:
- Scoring moves processor-side: v5 has no `Scores` key, so the `argmax` is computed
  here over the raw `scores` tensor (expected `[1, 5]` float32, per
  `tools/export_clip_prompt_scorer.py`) instead of being handed a named map by the
  engine.
- `event_cooldown_sec` is a `UserSettings` control (replacing v4's
  `externalprocessor.eventCooldown`), and the five prompts are a `UserSettings`
  `Repeater` (replacing v4's `externalprocessor.prompt*` keys), matching the
  paired preprocessor's dialog.
- The attribute key is the stable `clip_prompt` (the winning prompt text is the
  value), not the prompt text used as both key and value like v4: an
  operator-edited prompt text is not a stable attribute name.
- The emitted event carries the `clip.event` id registered in
  `external_postprocessors.json` verbatim: the AI Manager copies event ids into
  the DeviceAgent manifest unprefixed, unlike object classes.
"""

import os
import struct
import sys
import time
import logging
from collections import OrderedDict

script_location = os.path.dirname(os.path.abspath(__file__))
if os.path.join(script_location, "..") not in sys.path:
    sys.path.insert(0, os.path.join(script_location, ".."))

import nxai_postprocessor_sdk

logger = logging.getLogger(__name__)

PROCESSOR_NAME = "clip-example"
EVENT_ID_CLIP_PROMPT_RECOGNIZED = "clip.event"
CLIP_PROMPT_ATTRIBUTE = "clip_prompt"

SCORES_TENSOR_NAME = "scores"
PROMPT_COUNT = 5
EXPECTED_SCORES_SHAPE = [1, PROMPT_COUNT]
DEFAULT_EVENT_COOLDOWN_SEC = 0

CACHE_MAX_SIZE = 100

# Winning prompt per chained object, keyed by OriginalObjectID. Ported from v4's
# `objects_attributes` exactly: module-level, FIFO-evicted, no TTL.
objects_attributes = OrderedDict()
# Per-device event-cooldown state, ported from v4's `previous_state`/`previous_event_time`.
previous_state = {}
previous_event_time = {}

# ============================================================================
# Business Logic
# ============================================================================


def handle_message(msg: nxai_postprocessor_sdk.ObjectsPostProcessorMessage) -> nxai_postprocessor_sdk.ObjectsPostProcessorMessage:
    """Apply cached prompts to detector objects, score a chained CLIP frame, cache, emit.

    Args:
        msg: The message, parsed by `ObjectsPostProcessorMessage.from_dict()`.

    Returns:
        The same `msg`, with cached `clip_prompt` attributes applied to detector
        objects and, on a CLIP-scoring frame, a newly cached winning prompt and/or
        a deduplicated `Events` entry.
    """
    _apply_cached_prompts(msg)

    prompts = _read_prompts(msg.user_settings)
    event_cooldown_sec = _get_float_setting(
        msg.user_settings, "event_cooldown_sec", DEFAULT_EVENT_COOLDOWN_SEC
    )

    scores = _extract_scores_tensor(msg.tensors)
    if scores is None:
        return msg

    winner = _argmax_prompt(scores, prompts)
    if winner is None:
        return msg
    winning_prompt, _ = winner

    _maybe_emit_event(msg, winning_prompt, event_cooldown_sec)

    if msg.original_object_id is not None:
        _cache_winning_prompt(msg.original_object_id, winning_prompt)

    return msg


def run_message_loop(processor: nxai_postprocessor_sdk.ExternalProcessor) -> None:
    """Serve messages until the engine stops sending or the response channel fails."""
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

            msg = processor.parse_objects_message(message_bytes)
            if msg is None:
                logger.warning("Could not parse message, skipping")
                continue

            msg = handle_message(msg)

            response_bytes = processor.write_objects_message(msg)
            if response_bytes is None:
                logger.error("Failed to serialize message")
                continue
            if not processor.send_response(response_bytes):
                logger.error("Failed to send response")
                return
            logger.debug("Response sent successfully")
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            continue


# ============================================================================
# Private helpers
# ============================================================================


def _apply_cached_prompts(msg: nxai_postprocessor_sdk.ObjectsPostProcessorMessage) -> None:
    """Apply a previously cached winning prompt to each detector object, if any."""
    for class_objects in msg.objects.values():
        for obj in class_objects:
            cached_prompt = objects_attributes.get(obj.track_id)
            if cached_prompt is not None:
                obj.attributes[CLIP_PROMPT_ATTRIBUTE] = cached_prompt


def _read_prompts(user_settings: dict) -> dict:
    """Read `prompt1`..`prompt5` from UserSettings, excluding empty slots."""
    prompts = {}
    for index in range(1, PROMPT_COUNT + 1):
        text = user_settings.get(f"prompt{index}", "")
        if text:
            prompts[index] = text
    return prompts


def _get_float_setting(user_settings: dict, key: str, default: float) -> float:
    """Read a UserSettings value as a float, falling back to `default` on a bad value."""
    value = user_settings.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning(f"Invalid value for setting '{key}': {value!r}, using default {default}")
        return default


def _extract_scores_tensor(tensors: dict) -> "list[float] | None":
    """Decode the `scores` tensor out of a message's `msg.tensors` map.

    An empty `tensors` map or an absent `scores` entry is the normal case for a non-CLIP frame
    (e.g. the detector node) -> debug-log, return None. Present but the wrong
    shape/dtype is an operator/model-contract mismatch -> error-log expected vs.
    actual, return None. Present and matching the contract -> decoded floats.
    """
    if not tensors:
        logger.debug("No tensors on this message; not a CLIP-scoring frame")
        return None

    tensor = tensors.get(SCORES_TENSOR_NAME)
    if tensor is None:
        logger.debug(f"No '{SCORES_TENSOR_NAME}' tensor in message; not a CLIP-scoring frame")
        return None

    shape = tensor.shape
    data_type = tensor.data_type
    data = tensor.data
    expected_bytes = PROMPT_COUNT * 4

    if (
        shape != EXPECTED_SCORES_SHAPE
        or data_type != nxai_postprocessor_sdk.DATA_TYPE_FLOAT
        or len(data) != expected_bytes
    ):
        logger.error(
            f"'{SCORES_TENSOR_NAME}' tensor does not match the expected contract: "
            f"wanted shape {EXPECTED_SCORES_SHAPE} and data_type "
            f"{nxai_postprocessor_sdk.DATA_TYPE_FLOAT} (float), "
            f"got shape {shape} and data_type {data_type}"
        )
        return None

    return list(struct.unpack(f"<{PROMPT_COUNT}f", data))


def _argmax_prompt(scores: list, prompts: dict) -> "tuple[str, float] | None":
    """Pick the highest-scoring non-empty prompt slot. Empty slots never win."""
    best = None
    for index, text in prompts.items():
        score = scores[index - 1]
        if best is None or score > best[1]:
            best = (text, score)
    return best


def _maybe_emit_event(msg, winning_prompt: str, event_cooldown_sec: float) -> None:
    """Append a deduplicated `clip.event` for this device, respecting the cooldown."""
    device_id = msg.device_id
    now = time.time()

    if device_id in previous_state and previous_state[device_id] == winning_prompt:
        if (now - previous_event_time.get(device_id, 0)) < event_cooldown_sec:
            return

    event = nxai_postprocessor_sdk.EventMetadata()
    event.type_id = EVENT_ID_CLIP_PROMPT_RECOGNIZED
    event.description = winning_prompt
    msg.events.append(event)

    previous_event_time[device_id] = now
    previous_state[device_id] = winning_prompt


def _cache_winning_prompt(original_object_id: str, winning_prompt: str) -> None:
    """Cache the winning prompt for `original_object_id`, FIFO-evicting past 100 entries."""
    objects_attributes[original_object_id] = winning_prompt
    while len(objects_attributes) > CACHE_MAX_SIZE:
        objects_attributes.popitem(last=False)


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
