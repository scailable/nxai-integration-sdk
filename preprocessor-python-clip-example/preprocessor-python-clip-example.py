"""
CLIP Text Preprocessor (v5 SDK)

Tokenizes up to five text prompts forwarded via UserSettings and writes them
into the "text" tensor as CLIP token ids, so a chained CLIP model can score
the frame against the operator-configured prompts. The engine never builds a
"text" tensor of its own, so the tensor is appended when absent. All other
tensors pass through unchanged.
"""

import struct
import sys
import logging

import numpy as np
import instant_clip_tokenizer
import nxai_postprocessor_sdk

# ============================================================================
# Configuration
# ============================================================================

logger = logging.getLogger(__name__)

PROCESSOR_NAME = "clip-example"
TEXT_TENSOR_NAME = "text"
PROMPT_COUNT = 5
TOKEN_CONTEXT_LENGTH = 77

tokenizer = instant_clip_tokenizer.Tokenizer()

# ============================================================================
# Business Logic
# ============================================================================


def handle_message(msg: nxai_postprocessor_sdk.TensorMessage) -> nxai_postprocessor_sdk.TensorMessage:
    """Tokenize the forwarded prompts and overwrite the "text" tensor's data.

    Args:
        msg: Parsed TensorMessage from NXAI runtime.

    Returns:
        The same message, carrying a "text" tensor filled with the tokenized
        prompts. The tensor is appended when the engine did not supply it.
    """
    expected_shape = [PROMPT_COUNT, TOKEN_CONTEXT_LENGTH]

    tensor = msg.tensors.get(TEXT_TENSOR_NAME)
    if tensor is None:
        # The engine builds input tensors by manifest name and only ever
        # materializes "image-", "mask-" and "nms_sensitivity-", so the "text"
        # input a chained CLIP model declares never arrives. Append it
        # ourselves: the engine rebuilds the whole tensor array from our
        # response, so an added tensor does reach the runtime.
        logger.debug(f"No '{TEXT_TENSOR_NAME}' tensor in message; appending one")
        tensor = nxai_postprocessor_sdk.TensorMetadata()
        tensor.name = TEXT_TENSOR_NAME
        tensor.shape = expected_shape
        tensor.data_type = nxai_postprocessor_sdk.DATA_TYPE_INT32
        msg.tensors[TEXT_TENSOR_NAME] = tensor
    elif (
        list(tensor.shape) != expected_shape
        or tensor.data_type != nxai_postprocessor_sdk.DATA_TYPE_INT32
    ):
        # The [PROMPT_COUNT, TOKEN_CONTEXT_LENGTH] int32 shape is a contract
        # this example imposes on the chained model, not something it reads off
        # the tensor (see the README).
        # Writing our tokens into a differently shaped tensor would corrupt the
        # model's input silently, so refuse instead and say what was expected.
        logger.error(
            f"'{TEXT_TENSOR_NAME}' tensor does not match the contract this example expects: "
            f"wanted shape {expected_shape} and data_type "
            f"{nxai_postprocessor_sdk.DATA_TYPE_INT32} (int32), "
            f"got shape {list(tensor.shape)} and data_type {tensor.data_type}; "
            "leaving the tensor untouched"
        )
        return msg

    prompts = [msg.user_settings.get(f"prompt{i}", "") for i in range(1, PROMPT_COUNT + 1)]
    logger.debug(f"Tokenizing prompts: {prompts}")

    text_tokens = np.array(
        tokenizer.tokenize_batch(prompts, context_length=TOKEN_CONTEXT_LENGTH), dtype=np.int32
    )
    tensor.data = struct.pack(f"{text_tokens.size}i", *text_tokens.flatten())
    logger.debug(f"Wrote {text_tokens.size} tokens into '{TEXT_TENSOR_NAME}'")

    return msg


def run_message_loop(processor: nxai_postprocessor_sdk.ExternalPreprocessor) -> None:
    """Serve messages until the engine sends a graceful EXIT signal.

    `wait_for_message()` returns None both on a plain timeout and on an IPC EXIT
    signal; only `processor.exit_requested` tells them apart. A loop that just
    `continue`s on None regardless of that flag never returns, and the AI Manager
    kills the process after its grace period instead of it exiting cleanly.

    Args:
        processor: An initialized ExternalPreprocessor.
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
            tensor_message = processor.parse_tensor_message(message_bytes)
            if tensor_message is None:
                logger.error("Failed to parse message")
                continue
            processed_msg = handle_message(tensor_message)
            response_bytes = processor.write_tensor_message(processed_msg)
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
    logger.info(f"Initializing '{processor_config.get('ProcessorName', PROCESSOR_NAME)}' preprocessor")
    logger.debug(f"Command-line arguments: {sys.argv}")

    processor = nxai_postprocessor_sdk.ExternalPreprocessor()
    if not processor.initialize_from_args():
        logger.error("Failed to initialize preprocessor from arguments")
        sys.exit(1)
    logger.info(f"Preprocessor '{processor.name}' initialized successfully")
    logger.info("Waiting for messages from NXAI runtime...")
    try:
        run_message_loop(processor)
    except KeyboardInterrupt:
        logger.info("Exited with keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("Preprocessor shutdown")
