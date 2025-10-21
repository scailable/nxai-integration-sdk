import os
import sys
import socket
import signal
import logging
import logging.handlers
import msgpack
import configparser
import struct

# Add the nxai-utilities python utilities
script_location = os.path.dirname(sys.argv[0])
sys.path.append(os.path.join(script_location, "../nxai-utilities/python-utilities"))
import nxai_communication_utils

CONFIG_FILE = os.path.join(script_location, "..", "etc", "plugin.tensor.pre.ini")

# Set up logging
if os.path.exists(os.path.join(script_location, "..", "etc")):
    LOG_FILE = os.path.join(script_location, "..", "etc", "plugin.tensor.pre.log")
else:
    LOG_FILE = os.path.join(script_location, "plugin.tensor.pre.log")

# Initialize plugin and logging, script makes use of INFO and DEBUG levels
handler_stream = logging.StreamHandler(sys.stdout)
handler_stream.setLevel(logging.DEBUG)
handler_file = logging.FileHandler(filename=LOG_FILE, mode="w")
handler_file.setLevel(logging.DEBUG)

logging.basicConfig(format="%(asctime)s - %(levelname)s - example - %(message)s", handlers=[handler_stream, handler_file])

# The socket this preprocessor will listen on.
# This is always given as the first argument when the process is started
# But it can be manually defined as well, as long as it is the same as the socket path in the runtime settings
Preprocessor_Socket_Path = "/tmp/python-tensor-example-preprocessor.sock"

# Define a single SHM object to share images back to AI Manager
global output_shm
output_shm = None

global input_shm
input_shm = None


def parseTensorFromSHM(shm_key: int, external_settings: dict):
    global input_shm
    if input_shm is None:
        input_shm = nxai_communication_utils.SharedMemory(key=shm_key)
    ######### Get input tensor from SHM
    tensor_raw_data = input_shm.read()
    tensor_data = msgpack.unpackb(tensor_raw_data)

    if tensor_data is None or isinstance(tensor_data, dict) == False or "Tensors" not in tensor_data:
        logger.error("Invalid input tensor received. Ignoring.")
        return 0

    ######## Get nms setting ( if any )

    new_nms_value = 0.8
    if "externalprocessor.nmsoverride" in external_settings:
        try:
            new_nms_value = float(external_settings["externalprocessor.nmsoverride"])
        except:
            pass

    ######## Modify tensor as example

    for tensor_name, _ in tensor_data["Tensors"].items():
        logger.info("Got tensor name: " + str(tensor_name))
        if tensor_name == "nms_sensitivity-":
            # Set nms to new_nms_value
            tensor_data["Tensors"][tensor_name] = struct.pack("f", new_nms_value)

    ######## Write modified tensor to SHM

    output_data = msgpack.packb(tensor_data)

    global output_shm
    if output_shm is None:
        # Can reuse SHM ( if data is smaller or equal size ) or create new SHM and return ID
        output_data_size = len(output_data)
        output_shm = nxai_communication_utils.SharedMemory(size=output_data_size)
        logger.info("Created SHM with Key: " + output_shm.key)

    output_shm.write(output_data)

    return output_shm.key


def main():
    # Start socket listener to receive messages from NXAI runtime
    server = nxai_communication_utils.SocketListener(Preprocessor_Socket_Path)
    # Wait for messages in a loop
    while True:
        # Wait for input message from runtime
        try:
            connection, input_message = server.accept()
        except nxai_communication_utils.SocketTimeout:
            # Request timed out. Continue waiting
            continue

        tensor_header = msgpack.unpackb(input_message)
        print("EXAMPLE PREPROCESSOR: Received input message: ", tensor_header)

        external_settings = {}
        if "ExternalProcessorSettings" in tensor_header:
            logger.info("Got settings: " + str(tensor_header["ExternalProcessorSettings"]))
            external_settings = tensor_header["ExternalProcessorSettings"]

        # Process image
        output_shm_key = parseTensorFromSHM(tensor_header["SHMKEY"], external_settings)

        if output_shm_key != 0:
            tensor_header["SHMKEY"] = output_shm_key

        # Write header to respond
        output_message = msgpack.packb(tensor_header)

        # Send message back to runtime
        connection.send(output_message)
        connection.close()


def signalHandler(sig, _):
    print("EXAMPLE PREPROCESSOR: Received interrupt signal: ", sig)
    # Detach and destroy output shm
    if output_shm is not None:
        output_shm.detach()
        output_shm.remove()
    sys.exit(0)


def config():
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

    except Exception as e:
        logger.error(e, exc_info=True)

    logger.debug("Read configuration done")


def set_log_level(level):
    try:
        logger.setLevel(level)
    except Exception as e:
        logger.error(e, exc_info=True)


if __name__ == "__main__":
    print("EXAMPLE PREPROCESSOR: Input parameters: ", sys.argv)
    # Parse input arguments
    if len(sys.argv) > 1:
        Preprocessor_Socket_Path = sys.argv[1]
    # Handle interrupt signals
    signal.signal(signal.SIGTERM, signalHandler)

    ## initialize the logger
    logger = logging.getLogger(__name__)

    ## read configuration file if it's available
    config()

    logger.info("Initializing example plugin")
    logger.debug("Input parameters: " + str(sys.argv))

    try:
        main()
    except Exception as e:
        logger.error(e, exc_info=True)
    except KeyboardInterrupt:
        logger.info("Exited with keyboard interrupt")

    try:
        os.unlink(Preprocessor_Socket_Path)
    except OSError:
        if os.path.exists(Preprocessor_Socket_Path):
            logger.error("Could not remove socket file: " + Preprocessor_Socket_Path)
