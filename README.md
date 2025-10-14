NX Integration SDK
==================

The NX Integration SDK provides tools and examples for clients to create applications which seamlessly integrates with the NX software stack.

See the subdirectories for further documentation.

# Requirements

## Download the integration SDK

You probably have the integration SDK already if you're looking at this readme, the command to get the full integration SDK is as follows:

```shell
git clone https://github.com/scailable/sclbl-integration-sdk.git --recurse-submodules
```

If you have downloaded the sdk previously, you can also update to the latest version of the integration SDK while in the directory of the downloaded git repository.

```shell
git pull --recurse-submodules
```

The repository should be cloned with `--recurse-submodules`.

## Requirements to compile the pre/postprocessors

These applications can be compiled on any architecture natively in a Linux environment.

To compile, some software packages are required. These can be installed by running:

```shell
sudo apt install cmake
sudo apt install g++
```

For the python pre/postprocessors the following is also required

```shell
sudo apt install python3-pip
sudo apt install python3-venv
sudo apt install patchelf
```

## Requirements to run the pre/postprocessors

These applications can be run on any platform on which they can be compiled.

# How to use

## Getting Started

This project is CMake based, and all its modules can be compiled or gathered with CMake commands.

Because the different pre/postprocessors must be compiled for each hardware architecture this repository does not include pre-built binaries. All processors can be compiled manually.

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

Set up CMake configuration:

```shell
cmake ..
```

Build a single processor:

```
cmake --build build --target <processor-name>
```

Build all targets:

```shell
cmake --build .
```

This will build the default target, which includes the all the example applications that are active in the `CMakeLists.txt`.

It is possible to only run specific examples, refer to the readme files in the subdirectories of those examples for specific instructions.

## Install the pre/postprocessors

Before installing make sure the target directory is writable.

```shell
sudo chmod 777 /opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/postprocessors/
sudo chmod 777 /opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/preprocessors/
```

To install a single processor to the default folder:
```
cmake --install build --component <processor-name>
```

To install the generated pre/postprocessor examples to the default pre/postprocessors folder:

```shell
cmake --build . --target install
```

## Defining the pre/postprocessors

Create a configuration file at `/opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/postprocessors/external_postprocessors.json` or `/opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/preprocessors/external_preprocessors.json` and add the details of your pre/postprocessors to the root object of that file. For example the following file enables *all* python based pre/postprocessors:

```json
{
    "externalPostprocessors": [
        {
            "Name":"EI-Upload-Postprocessor",
            "Command":"/opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/postprocessors/postprocessor-python-edgeimpulse-example",
            "SocketPath":"/tmp/python-edgeimpulse-postprocessor.sock",
            "ReceiveInputTensor": true,
            "RunLast": false,
            "NoResponse": true
        },
        {
            "Name":"Example-Postprocessor",
            "Command":"/opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/postprocessors/postprocessor-python-example",
            "SocketPath":"/tmp/python-example-postprocessor.sock",
            "ReceiveInputTensor": false
        },
        {
            "Name":"Image-Postprocessor",
            "Command":"/opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/postprocessors/postprocessor-python-image-example",
            "SocketPath":"/tmp/python-image-postprocessor.sock",
            "ReceiveInputTensor": true
        },
        {
            "Name":"NoResponse-Postprocessor",
            "Command":"/opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/postprocessors/postprocessor-python-noresponse-example",
            "SocketPath":"/tmp/python-noresponse-postprocessor.sock",
            "ReceiveInputTensor": false,
            "ReceiveBinaryData": false,
            "NoResponse": true
        },
        {
            "Name":"Cloud-Inference-Postprocessor",
            "Command":"/opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/postprocessors/postprocessor-cloud-inference-example",
            "SocketPath":"/tmp/python-cloud-inference-postprocessor.sock",
            "ReceiveInputTensor": true
        }

    ]
}
```

``` json
{
    "externalPreprocessors": [
        {
            "Name":"Example-Preprocessor",
            "Command":"/opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/preprocessors/preprocessor-python-example",
            "SocketPath":"/tmp/example-preprocessor.sock"
        }
    ]
}
```

## Restarting the server

Finally, to (re)load your new pre/postprocessor, make sure to restart the NX Server with:

```shell
sudo service networkoptix-metavms-mediaserver restart
```

You also want to make sure the pre/postprocessor can be used by the NX AI Manager (this is the mostly same command as earlier)

```shell
sudo chmod -R a+x /opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/postprocessors/
sudo chmod -R a+x /opt/networkoptix-metavms/mediaserver/bin/plugins/nxai_plugin/nxai_manager/preprocessors/
```

## Selecting to the pre/postprocessor

If the pre/postprocessor is defined correctly, its name should appear in the list of pre/postprocessors in the NX Cloud Pipelines UI. If it is selected in the pipeline settings then the NxAI Manager will send data to the pre/postprocessor and wait for its output.

# Development

For fast iterative development it is often desired not to fully recompile the postprocessor to have it reloaded. 

Functionality has therefore been added to instruct the AI Manager not to start the external postprocessor, but instead assume it's running.
The AI Manager will then attempt to send data to the external processor each frame, and wait for a response.
You can then start the processor yourself to immediately see its effect.
You can also print/manipulate data easily to get familiar with the format of the data that the processor receives. 

Once you are satisfied with how your processor behaves, you can then compile and package it.

To instruct the AI Manager to not start the external processor, simply omit the "Command" field in the processor definition.

# Directory structure

```
.
├── CMakeLists.txt
├── nxai-utilities
│   ├── CMakeLists.txt
│   ├── include
│   │   ├── mpack.h
│   │   ├── musl_time.h
│   │   ├── nxai_data_structures.h
│   │   ├── nxai_data_utils.h
│   │   ├── nxai_process_utils.h
│   │   ├── nxai_shm_utils.h
│   │   ├── nxai_socket_utils.h
│   │   └── yyjson.h
│   ├── python-utilities
│   │   ├── communication_utils.py
│   │   └── requirements.txt
│   ├── README.md
│   └── src
│       ├── mpack.c
│       ├── nxai_data_utils.c
│       ├── nxai_process_utils.c
│       ├── nxai_shm_utils.c
│       ├── nxai_socket_utils.c
│       └── yyjson.c
├── postprocessor-c-example
│   ├── CMakeLists.txt
│   ├── deps
│   │   ├── data_utils.h
│   │   ├── mpack.c
│   │   └── mpack.h
│   ├── README.md
│   └── src
│       ├── data_utils.c
│       └── main.c
├── postprocessor-c-image-example
│   ├── CMakeLists.txt
│   ├── deps
│   │   ├── data_utils.h
│   │   ├── mpack.c
│   │   └── mpack.h
│   ├── README.md
│   └── src
│       ├── data_utils.c
│       └── main.c
├── postprocessor-cloud-inference-example
│   ├── aws_utils.py
│   ├── CMakeLists.txt
│   ├── plugin.cloud-inference.ini.example
│   ├── postprocessor-cloud-inference-example.py
│   ├── README.md
│   └── requirements.txt
├── postprocessor-c-raw-example
│   ├── CMakeLists.txt
│   ├── deps
│   │   ├── data_utils.h
│   │   ├── mpack.c
│   │   └── mpack.h
│   ├── README.md
│   └── src
│       ├── data_utils.c
│       └── main.c
├── postprocessor-python-confidences-example
│   ├── CMakeLists.txt
│   ├── plugin.confidences.ini.example
│   ├── postprocessor-python-confidences-example.py
│   ├── README.md
│   └── requirements.txt
├── postprocessor-python-edgeimpulse-example
│   ├── CMakeLists.txt
│   ├── plugin.edgeimpulse.ini.example
│   ├── postprocessor-python-edgeimpulse-example.py
│   ├── README.md
│   └── requirements.txt
├── postprocessor-python-events-example
│   ├── CMakeLists.txt
│   ├── plugin.events.ini.example
│   ├── postprocessor-python-events-example.py
│   ├── README.md
│   └── requirements.txt
├── postprocessor-python-example
│   ├── CMakeLists.txt
│   ├── plugin.example.ini.example
│   ├── postprocessor-python-example.py
│   ├── README.md
│   └── requirements.txt
├── postprocessor-python-image-example
│   ├── CMakeLists.txt
│   ├── plugin.image.ini.example
│   ├── postprocessor-python-image-example.py
│   ├── README.md
│   └── requirements.txt
├── postprocessor-python-noresponse-example
│   ├── CMakeLists.txt
│   ├── plugin.noresponse.ini.example
│   ├── postprocessor-python-noresponse-example.py
│   ├── README.md
│   └── requirements.txt
├── postprocessor-python-settings-example
│   ├── CMakeLists.txt
│   ├── plugin.settings.ini.example
│   ├── postprocessor-python-settings-example.py
│   ├── README.md
│   ├── requirements.txt
│   └── settings_model.md
├── preprocessor-python-example
│   ├── CMakeLists.txt
│   ├── preprocessor-python-example.py
│   ├── README.md
│   └── requirements.txt
└── README.md
```

# Licence

Copyright 2025, Network Optix, All rights reserved.