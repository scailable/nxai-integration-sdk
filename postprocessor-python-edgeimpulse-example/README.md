Edge Impulse Image Uploader
===========================

This example application uses a Python based postprocessor to upload images to Edge Impulse for training, either interval based with the auto uploader, or based on an inference confidence threshold.

# Requirements

For this example to work, you can use any model.

You also need an Edge Impulse account at https://edgeimpulse.com/.

Edge Impulse usage charges may apply, but it is possible to use the free tier to test this.

# How to use

## Download the integration SDK

You probably have the integration SDK already if you're looking at this readme, the command to get the full integration SDK is as follows:

```shell
git clone https://github.com/scailable/sclbl-integration-sdk.git --recurse-submodules
```

If you have downloaded the sdk previously, you can also update to the latest version of the integration SDK while in the directory of the downloaded git repository.

```shell
git pull --recurse-submodules
```

## Configuration of example postprocessor

Create a new [configuration file](plugin.edgeimpulse.ini.example) at `/opt/networkoptix-metavms/mediaserver/bin/plugins/nx_ai_manager_plugin/nxai_manager/etc/plugin.edgeimpulse.ini` and add your Edge Impulse API key as well as any overrides of the following settings.

```ini
[common]
debug_level=INFO
[edgeimpulse]
# Add your own project level Edge Impulse API key
api_key = ei_add_your_key_here
# Option autogenerate images every x seconds as an alternative to sending based on p_value
auto_generator = True
# If auto_generator True, every how many seconds upload an image?
auto_generator_every_seconds = 1
# Flush the buffer at this length
samples_buffer_flush_size = 20
# Send images below this value to EdgeImpulse. Can be between 0.0 and 1.0
p_value = 0.4
```

Before building this postprocessor you need to enter your API key for the Edge Impulse Project.

Replace the key in `api_key` with your own key, you can get your key in Edge Impulse Studio from the "Dashboard > Keys" page. Don't use quotes around the key in the configuration file.

## Use the interval based upload

When the `auto_generator` is set to `True` images will be uploaded according to the value in `auto_generator_every_seconds`

## Use the confidence threshold based upload

When the `auto_generator` is set to `False` images will be uploaded according to the value in `p_value`

## Preparation of dependencies

Install the needed dependencies

```shell
sudo apt install cmake
sudo apt install g++
sudo apt install python3-pip
sudo apt install patchelf
```

Change into the directory created for the project if you're not already there.

```shell
cd sclbl-integration-sdk/
```

Prepare the *build* directory in the project directory, and switch to the build directory.

```shell
mkdir -p build
cd build
```

Set up a python virtual environment in the newly created build directory (on recent ubuntu servers this is required).

```shell
python3 -m venv integrationsdk
source integrationsdk/bin/activate
```

## (optionally) Remove other postprocessors for compilation

Edit the `CMakelist.txt` to disable all but the external postprocessor.

```shell
nano ../CMakeLists.txt
```

It should look similar to this

```shell
cmake_minimum_required(VERSION 3.10.2)

project(sclbl-integration-examples)

# Add Scailable C Utilities for all subprojects
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/sclbl-utilities)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/sclbl-utilities/include)

# Add Edge Impulse Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-python-edgeimpulse-example)

# Add installation option
install(TARGETS
    DESTINATION /opt/networkoptix-metavms/mediaserver/bin/plugins/nx_ai_manager_plugin/nxai_manager/postprocessors/
)
install(PROGRAMS
    ${CMAKE_CURRENT_BINARY_DIR}/postprocessor-python-edgeimpulse-example/postprocessor-python-edgeimpulse-example
    DESTINATION /opt/networkoptix-metavms/mediaserver/bin/plugins/nx_ai_manager_plugin/nxai_manager/postprocessors/
)
```

## Compile the postprocessor

Build the postprocessor, while in the created *build* directory. This may take a while, depending on the speed of your system.

```shell
cmake ..
make
```

## Install the postprocessor

Once compiled, copy the executable to an accessible directory.

A convenience directory within the Edge AI Manager installation is created for this purpose at `/opt/networkoptix-metavms/mediaserver/bin/plugins/nx_ai_manager_plugin/nxai_manager/postprocessors`.

The application and settings files you add must be readable and executable by the NX AI Edge AI Manager. This can be achieved by running:

```
sudo chmod -R 777 /opt/networkoptix-metavms/mediaserver/bin/plugins/nx_ai_manager_plugin/nxai_manager/postprocessors
```

Install the postprocessor automatically with the cmake command, also from within the *build* directory.

```shell
cmake --build . --target install
```

## Defining the postprocessor

To make the postprocessor available in the NX Server another configuration must be added.

Create a configuration file at `/opt/networkoptix-metavms/mediaserver/bin/plugins/nx_ai_manager_plugin/nxai_manager/postprocessors/external_postprocessors.json` and add the details of your postprocessor to the root object of that file. For example:

``` json
{
    "externalPostprocessors": [
        {
            "Name":"Example-Postprocessor",
            "Command":"/opt/networkoptix-metavms/mediaserver/bin/plugins/nx_ai_manager_plugin/nxai_manager/postprocessors/postprocessor-python-edgeimpulse-example",
            "SocketPath":"/tmp/python-edgeimpulse-postprocessor.sock",
            "ReceiveInputTensor": true,
            "RunLast": false,
            "NoResponse": true
        }
    ]
}
```

This tells the Edge AI Manager about the postprocessor:

- **Name** gives the postprocesor a name so it can be selected later
- **Command** defines how to start the postprocessor
- **SocketPath** tells the AI Manager where to send data to so the external postprocessor will receive it
- **ReceiveInputTensor** tells the AI Manager if this postprocessor expects information to access the raw input tensor data
- **Runlast** indicates when the postprocessor is run. When 'RunLast' is set to false, the external post-processor runs immediately after the model's inference. This means the raw data sent to the post-processor includes the output tensors' names, data, shapes, and data types. If 'RunLast' is set to true or left out, the external post-processor runs later, after the built-in post-processors. In this scenario, the data sent to the external post-processor is already processed, including output scores and bounding boxes (if applicable), scaled to match the original image dimensions rather than the model's processed image.
- **NoResponse** tells the AI Manager to not wait for a response from this postprocessor

The socket path is always given as the first command line argument when the application is started. It is therefore best practice for the external postprocessor application to read its socket path from here, instead of defining the data twice.

## Restarting the server

Finally, to (re)load your new postprocessor, make sure to restart the NX Server with:

```shell
sudo service networkoptix-metavms-mediaserver restart
```

You also want to make sure the postprocessor can be used by the NX AI Manager (this is the mostly same command as earlier)

```shell
sudo chmod -R a+x /opt/networkoptix-metavms/mediaserver/bin/plugins/nx_ai_manager_plugin/nxai_manager/postprocessors/
```

## Selecting to the postprocessor

If the postprocessor is defined correctly, its name should appear in the list of postprocessors in the NX Plugin settings. If it is selected in the plugin settings then the Edge AI Runtime will send data to the postprocessor and wait for its output.

## Output logging

There is an output log where the uploads can be tracked in real time from the server.

```shell
tail -f /opt/networkoptix-metavms/mediaserver/bin/plugins/nx_ai_manager_plugin/nxai_manager/etc/plugin.edgeimpulse.log
```

# Licence

Copyright 2025, Network Optix, All rights reserved.