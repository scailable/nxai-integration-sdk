import os
import sys
import socket
import logging
import logging.handlers
import configparser
from pprint import pformat
from PIL import Image
import msgpack
import struct
import numpy as np
from aws_utils import classify_faces, create_session

# Add the nxai-utilities python utilities
script_location = os.path.dirname(sys.argv[0])
sys.path.append(os.path.join(script_location, "../nxai-utilities/python-utilities"))
import nxai_communication_utils

CONFIG_FILE = os.path.join(script_location, "..", "etc", "plugin.cloud-inference.ini")

# Set up logging
if os.path.exists(os.path.join(script_location, "..", "etc")):
    LOG_FILE = os.path.join(script_location, "..", "etc", "plugin.cloud-inference.log")
else:
    LOG_FILE = os.path.join(script_location, "plugin.cloud-inference.log")

# Initialize plugin and logging, script makes use of INFO and DEBUG levels
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - cloud inference - %(message)s",
    filename=LOG_FILE,
    filemode="w",
)

# The name of the postprocessor.
# This is used to match the definition of the postprocessor with routing.
Postprocessor_Name = "Cloud-Inference-Postprocessor"

# The socket this postprocessor will listen on.
# This is always given as the first argument when the process is started
# But it can be manually defined as well, as long as it is the same as the socket path in the runtime settings
import tempfile
Postprocessor_Socket_Path = os.path.join(tempfile.gettempdir(),"python-cloud-inference-postprocessor.sock")

global shared_memory
shared_memory = None


def parse_image_from_shm(shm_key: int, width: int, height: int, channels: int):
    global shared_memory
    try:
        if shared_memory is None:
            shared_memory = nxai_communication_utils.SharedMemory(key=shm_key)
        image_data = shared_memory.read()
        image_size = width * height * channels
        image_array = list(struct.unpack("B" * image_size, image_data))
        image_array = np.array(image_array).reshape((height, width, channels)).astype("uint8")
    except Exception as e:
        logger.debug("Failed to parse image from shared memory: ", e)
        return None

    return image_array


def config():

    global aws_access_key_id
    global aws_secret_access_key
    global region_name
    global image_path

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

        aws_access_key_id = configuration.get("cloud", "aws_access_key_id", fallback=False)
        aws_secret_access_key = configuration.get("cloud", "aws_secret_access_key", fallback=False)
        region_name = configuration.get("cloud", "region_name", fallback=False)
        image_path = configuration.get(
            "inference",
            "image_path",
            fallback="/opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/nxai_manager/postprocessors/face.png",
        )

    except Exception as e:
        logger.error(e, exc_info=True)

    logger.debug("Read configuration done")


def set_log_level(level):
    try:
        logger.setLevel(level)
    except Exception as e:
        logger.error(e, exc_info=True)


def main():

    global aws_access_key_id
    global aws_secret_access_key
    global region_name
    global image_path

    # Start socket listener to receive messages from NXAI runtime
    server = nxai_communication_utils.SocketListener(Postprocessor_Socket_Path)
    # Wait for messages in a loop
    while True:
        # Wait for input message from runtime
        logger.debug("Waiting for input message")

        try:
            connection, input_message = server.accept()
        except nxai_communication_utils.SocketTimeout:
            # Request timed out. Continue waiting
            continue

        # Since we're also expecting an image, receive the image header
        try:
            image_header = connection.receive()
        except nxai_communication_utils.SocketTimeout:
            # Did not receive image header
            logger.debug("Did not receive image header. Are the settings correct?")
            continue

        # Parse input message
        input_object = nxai_communication_utils.parseInferenceResults(input_message)
        if isinstance(input_object, nxai_communication_utils.ExitSignal):
            logger.info("Received exit signal.")
            connection.close()
            break

        image_header = msgpack.unpackb(image_header)
        image_array = parse_image_from_shm(
            image_header["SHMKEY"],
            image_header["Width"],
            image_header["Height"],
            image_header["Channels"],
        )
        if image_array is None:
            continue

        image = Image.fromarray(image_array)

        faces = np.array(input_object["BBoxes_xyxy"]["face"]).reshape(-1, 4)

        faces_to_delete = []
        for i, face in enumerate(faces):
            path = image_path
            x1, y1, x2, y2 = face
            cropped_image = image.crop((x1, y1, x2, y2))
            cropped_image.save(path)

            logger.info("Classifying image " + path)

            description = classify_faces(path, logger)

            if description is None:
                logger.info("No description for this face.")
                continue

            # Add the description to the object
            if description not in input_object:
                input_object["BBoxes_xyxy"][description] = face.tolist()
            else:
                input_object["BBoxes_xyxy"][description].extend(face.tolist())

            faces_to_delete.append(i)

            # FIXME: Run the classification for 2 faces max to avoid affecting FPS rate
            if len(faces_to_delete) >= 2:
                break

        # Delete the faces that have been classified
        faces = np.delete(faces, faces_to_delete, axis=0)
        input_object["BBoxes_xyxy"]["face"] = faces.flatten().tolist()

        formatted_packed_object = pformat(input_object)
        logger.debug(f"Returning packed object:\n\n{formatted_packed_object}\n\n")

        # Write object back to string
        output_message = nxai_communication_utils.writeInferenceResults(input_object)

        # Send message back to runtime
        connection.send(output_message)
        connection.close()


if __name__ == "__main__":
    ## initialize the logger
    logger = logging.getLogger(__name__)

    ## read configuration file if it's available
    config()

    logger.info("Initializing cloud interference plugin")
    logger.debug("Input parameters: " + str(sys.argv))

    global rekognition_client

    try:
        rekognition_client = create_session(logger)
    except Exception as e:
        logger.error(e, exc_info=True)

    if rekognition_client:
        logger.debug("AWS Session started")
    else:
        logger.error("AWS session failed")
        exit()

    # Parse input arguments
    if len(sys.argv) > 1:
        Postprocessor_Socket_Path = sys.argv[1]

    # Start program
    try:
        main()
    except Exception as e:
        logger.error(e, exc_info=True)
    except KeyboardInterrupt:
        logger.info("Exited with keyboard interrupt")

    try:
        os.unlink(Postprocessor_Socket_Path)
    except OSError:
        if os.path.exists(Postprocessor_Socket_Path):
            logger.error("Could not remove socket file: " + Postprocessor_Socket_Path)
