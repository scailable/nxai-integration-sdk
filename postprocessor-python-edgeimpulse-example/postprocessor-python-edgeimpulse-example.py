import os
import sys
import socket
import logging
import logging.handlers
import configparser
import io
import time
from pprint import pformat
import msgpack
import struct
from math import prod
from datetime import datetime
from PIL import Image
import edgeimpulse

script_location = os.path.dirname(sys.argv[0])
CONFIG_FILE = os.path.join(script_location, "..", "etc", "plugin.cloud-edgeimpulse.ini")

LOG_FILE = os.path.join(script_location, "..", "etc", "plugin.cloud-edgeimpulse.log")

# Add your own project level Edge Impulse API key
# Please use the CONFIG_FILE to set [edgeimpulse][api_key] to your edge impulse api key
default_edge_impulse_api_key = "ei_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# We keep a counter to generate unique filenames.
samples_counter: int = 0

# The buffer with samples to batch.
samples_buffer: list = []

# Returning data to the AI Manager is not needed for this postprocessor.
# See also "NoResponse": true value in external_postprocessors.json / README.md
return_data = False

# Initialize plugin and logging, script makes use of INFO and DEBUG levels
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - edge impulse - %(message)s",
    filename=LOG_FILE,
    filemode="w",
)

# Add the nxai-utilities python utilities
sys.path.append(os.path.join(script_location, "../nxai-utilities/python-utilities"))
import nxai_communication_utils

# The name of the postprocessor.
# This is used to match the definition of the postprocessor with routing.
Postprocessor_Name = "Python-EdgeImpulse-Postprocessor"

# The socket this postprocessor will listen on.
# This is always given as the first argument when the process is started
# But it can be manually defined as well, as long as it is the same as the socket path in the runtime settings
Postprocessor_Socket_Path = "/tmp/python-edgeimpulse-postprocessor.sock"


def send_samples_buffer():
    # This function sends the buffered samples to an Edge Impulse instance for data processing.
    # It generates a unique filename for each sample and adds it to a new list of samples while
    # updating the global counter.
    # It then uploads all these samples to the Edge Impulse. The function also times the duration of
    # the upload and prints this time along with the number of samples uploaded. If the sample buffer
    # was already empty, the function just prints that there were no samples to upload.
    # At the end, it empties the sample buffer.
    global samples_buffer, samples_counter
    if len(samples_buffer) > 0:
        logging.info("Sending {c} samples to Edge Impulse...".format(c=len(samples_buffer)))
        start_at = time.perf_counter()
        samples = []
        for contents in samples_buffer:
            samples_counter += 1
            logging.info("Create sample" + str(samples_counter))
            filename = "{dt}C{c}.jpg".format(dt=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), c=samples_counter)
            output = io.BytesIO(contents)
            sample = edgeimpulse.experimental.data.Sample(
                filename=filename,
                data=output,
                metadata={
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
            )
            samples.append(sample)

        logging.info("Done creating samples, starting upload...")
        response = edgeimpulse.experimental.data.upload_samples(samples)
        logging.info("Done upload")

        # Check to make sure there were no failures
        if (len(response.fails)) != 0:
            logging.info("Could not upload files")

        end_at = time.perf_counter()
        logging.info(
            "Send {c} samples in {d:0.1f}sec to Edge Impulse. Total {t}".format(
                c=len(samples_buffer),
                d=end_at - start_at,
                t=samples_counter,
            )
        )
        logging.info("Send a total of {t} samples to Edge Impulse".format(t=samples_counter))

        samples_buffer = []
    else:
        logging.info("No samples to send to Edge Impulse. Total {t}".format(t=samples_counter))


def config():

    global edge_impulse_api_key
    global default_edge_impulse_api_key
    global auto_generator
    global auto_generator_every_seconds
    global samples_buffer_flush_size
    global p_value

    logger.info("Reading configuration from:" + CONFIG_FILE)

    try:
        configuration = configparser.ConfigParser()
        configuration.read(CONFIG_FILE)

        configured_log_level = configuration.get("common", "debug_level", fallback="INFO")
        set_log_level(configured_log_level)

        for section in configuration.sections():
            logger.info("config section: " + section)
            for key in configuration[section]:
                logger.info("config key: " + key + " = " + configuration[section][key])

        # Override default values from config
        edge_impulse_api_key = configuration.get("edgeimpulse", "api_key", fallback=default_edge_impulse_api_key)

        logger.info("new edge_impulse_api_key: " + edge_impulse_api_key)

        auto_generator = configuration.get("edgeimpulse", "auto_generator", fallback=False)
        auto_generator_every_seconds = int(configuration.get("edgeimpulse", "auto_generator_every_seconds", fallback=1))
        samples_buffer_flush_size = int(configuration.get("edgeimpulse", "samples_buffer_flush_size", fallback=20))
        p_value = float(configuration.get("edgeimpulse", "p_value", fallback=0.4))

    except Exception as e:
        logger.error(e, exc_info=True)

    logger.debug("Read configuration done")


def set_log_level(level):
    try:
        logger.setLevel(level)
    except Exception as e:
        logger.error(e, exc_info=True)


def main():

    global samples_buffer
    global samples_buffer_flush_size
    global samples_counter
    global p_value
    global auto_generator
    global auto_generator_every_seconds
    global edge_impulse_api_key

    # Add your own project level Edge Impulse API key
    edgeimpulse.API_KEY = edge_impulse_api_key

    # Start socket listener to receive messages from NXAI runtime
    try:
        os.remove(Postprocessor_Socket_Path)
    except OSError:
        pass
    server = nxai_communication_utils.SocketListener(Postprocessor_Socket_Path)

    # Get the current time at the start
    start_time = time.time()

    # Wait for messages in a loop
    counter = 0
    while True:

        upload_sample = False

        logger.debug("Wait for message " + str(counter))
        # Wait for input message from runtime
        try:
            connection, input_message = server.accept()
            image_header = connection.receive()
        except nxai_communication_utils.SocketTimeout:
            # Request timed out. Continue waiting
            continue

        counter = counter + 1

        logger.debug("Message " + str(counter) + "received")

        # Parse input message
        parsed_response = msgpack.unpackb(input_message)

        # Read Output types, shapes and sizes
        output_data_types = parsed_response.get("OutputDataTypes")  # 1 for float32 and 3 for int8
        output_shapes = parsed_response.get("OutputShapes")
        output_sizes = [prod(output_shapes[i]) for i in range(len(output_shapes))]

        # Unpack Output values
        for i, key in enumerate(parsed_response["Outputs"]):
            value = parsed_response["Outputs"][key]
            if output_data_types[i] == 1:
                parsed_response["Outputs"][key] = list(struct.unpack("f" * output_sizes[i], value))
            elif output_data_types[i] == 3:
                parsed_response["Outputs"][key] = list(struct.unpack("b" * output_sizes[i], value))

        if logging.getLogger().isEnabledFor(logging.DEBUG):
            logger.debug("Message " + str(counter) + " parsed")
            # Use pformat to format the deep object
            formatted_object = pformat(parsed_response)
            logger.debug(f"Parsed response:\n\n{formatted_object}\n\n")

        current_time = time.time()

        # Check if auto_generator is True and 60 seconds have passed
        if auto_generator and current_time - start_time >= auto_generator_every_seconds:

            start_time = current_time
            logging.info("Add timed sample every " + str(auto_generator_every_seconds) + " seconds number " + str(counter) + " to upload queue")
            upload_sample = True

        elif not auto_generator:

            # Retrieve the bounding box values
            bbox_values = list(parsed_response["Outputs"].values())[0]

            # Number of elements in each bounding box entry (assuming format: x1, y1, x2, y2, score, class)
            num_elements_per_entry = 6

            # Extract every 5th out of six values
            parsed_values = [bbox_values[i] for i in range(4, len(bbox_values), num_elements_per_entry)]

            # Check if any of the values are below p_value and earmark result for retrieval
            for value in parsed_values:
                if value < p_value:
                    logger.debug("Parsed value: %.8f", value)
                    upload_sample = True

        if upload_sample:
            logger.debug("uploading sample")
            # Parse image information
            image_header = msgpack.unpackb(image_header)

            # Read image
            shared_memory = nxai_communication_utils.SharedMemory(key=image_header["SHMKey"])
            image_data = shared_memory.read()
            with Image.frombytes("RGB", (image_header["Width"], image_header["Height"]), image_data) as image:
                with io.BytesIO() as output:
                    image.save(output, format="JPEG")
                    output.seek(0)
                    samples_buffer.append(output.getvalue())
            if len(samples_buffer) >= samples_buffer_flush_size:
                send_samples_buffer()
        else:
            logger.debug("skipping sample")

        if return_data:
            # Create msgpack formatted message
            data_types = parsed_response.get("OutputDataTypes")
            for key in parsed_response["Outputs"]:
                value = parsed_response["Outputs"][key]
                if data_types[0] == 1:
                    parsed_response["Outputs"][key] = struct.pack("f" * len(value), *value)
                elif data_types[0] == 3:
                    parsed_response["Outputs"][key] = struct.pack("b" * len(value), *value)
            message_bytes = msgpack.packb(parsed_response)

            # Send message back to runtime
            connection.send(message_bytes)


if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    config()

    logger.info("Initializing example plugin")
    logger.debug("Input parameters: " + str(sys.argv))

    if edge_impulse_api_key == default_edge_impulse_api_key:
        logger.error("Edge Impulse Key is not set yet", exc_info=True)
        exit()
    else:
        logger.debug("Edge Impulse Key: " + edge_impulse_api_key)

    # Parse input arguments
    if len(sys.argv) > 1:
        Postprocessor_Socket_Path = sys.argv[1]

    # Start program
    try:
        main()
    except Exception as e:
        logger.error(e, exc_info=True)
