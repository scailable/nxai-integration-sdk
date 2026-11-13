"""
Minimal fixture postprocessor for the e2e harness's skipFeatureExtraction wire-contract
proof.

Not a general SDK example -- do not add it to CLAUDE.md's "Picking an example" list.
``skipFeatureExtraction`` is an engine-side chaining-gate field. It is opt-out and
defaults to ``false``: the engine chains every object unless a postprocessor sets the
field to ``true`` on that frame.

On every frame it emits two "chain-test" objects, regardless of what it received:
  - one with ``skip_feature_extraction = True`` (the gated object)
  - one that leaves the attribute unset, so ``to_dict()`` omits the key entirely (the
    ungated object; the real engine's deserializer reads an absent field as ``false``
    every frame)
"""

import sys
import logging
import msgpack
import nxai_postprocessor_sdk

logger = logging.getLogger(__name__)

PROCESSOR_NAME = "python-skipfeatureextraction-fixture"


def build_response_bytes(msg: nxai_postprocessor_sdk.ObjectsPostProcessorMessage) -> bytes:
    """Build the raw msgpack response carrying one gated and one ungated object."""
    objects_dict = {
        class_name: [obj.to_dict() for obj in objects_list]
        for class_name, objects_list in msg.objects.items()
    }

    gated = nxai_postprocessor_sdk.ObjectMetadata()
    gated.x, gated.y, gated.width, gated.height = 0.1, 0.1, 0.2, 0.2
    gated.confidence = 0.9
    gated.attributes["role"] = "gated"
    gated.skip_feature_extraction = True
    gated_dict = gated.to_dict()

    ungated = nxai_postprocessor_sdk.ObjectMetadata()
    ungated.x, ungated.y, ungated.width, ungated.height = 0.5, 0.5, 0.2, 0.2
    ungated.confidence = 0.9
    ungated.attributes["role"] = "ungated"
    # skip_feature_extraction stays None, so to_dict() omits the key: absence means
    # the engine chains the object.
    ungated_dict = ungated.to_dict()

    objects_dict["chain-test"] = [gated_dict, ungated_dict]

    message_dict = {
        "Timestamp": msg.timestamp,
        "DeviceID": msg.device_id,
        "DeviceName": msg.device_name,
        "Objects": objects_dict,
        "Events": [evt.to_dict() for evt in msg.events],
    }
    return msgpack.packb(message_dict, use_bin_type=True)


if __name__ == "__main__":
    logger = nxai_postprocessor_sdk.setup_logging()
    processor_config = nxai_postprocessor_sdk.load_config(sys.argv)
    nxai_postprocessor_sdk.apply_config_logging(processor_config, logger)

    processor = nxai_postprocessor_sdk.ExternalPostprocessor()
    if not processor.initialize_from_args():
        logger.error("Failed to initialize postprocessor from arguments")
        sys.exit(1)

    try:
        while True:
            message_bytes = processor.wait_for_message(timeout_ms=5000)
            if message_bytes is None:
                if processor.exit_requested:
                    logger.info("Received graceful exit signal, shutting down")
                    break
                continue

            objects_message = processor.parse_objects_message(message_bytes)
            if objects_message is None:
                logger.error("Failed to parse message")
                continue

            response_bytes = build_response_bytes(objects_message)
            if not processor.send_response(response_bytes):
                logger.error("Failed to send response")
                break
    except KeyboardInterrupt:
        logger.info("Exited with keyboard interrupt")
    finally:
        logger.info("Postprocessor shutdown")
