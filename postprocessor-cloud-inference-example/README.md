Cloud inference with the Nx AI Manager
=========================

This example is meant to be used to illustrate how the Nx AI Manager allows the user to perform additional data analysis on the stream logic on a cloud machine.
For the sake of example, the AWS Rekognition API was used to perform age, gender and emotion classification on faces detected by the AI Manager.

The workflow is as follows:
1. The AI Manager runs a face detection model on each frame
2. When the model inference is accomplished, the AI Manager sends its output to the Python external post-processor that sends a request (per detected face) to AWS Rekognition, and waits for response.
3. The post-processor overwrites the category of the detected object (ie. "face") to a more insightful one, such as: "A male in his 30s feeling happy", and sends back the result to Nx to visualize it.

# Requirements

For this example to work, you need to assign the Face locator model to your camera device. 

You also need an AWS account and the associated Access Key and Access Secret. Refer to the amazon AWS documentation on how to get these keys.
The user needs `AmazonRekognitionReadOnlyAccess` to use the features in this demo.

Amazon usage charges may apply, but it is possible to use the free tier to test this.

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

There are two configuration files you need to set for Amazon AWS to run this demo. 

You need to set your AWS credential keys in the configuration file at `~/.aws/credentials`:

```ini
[default]
aws_access_key_id=acced_key
aws_secret_access_key=secret_access_key
```

Also set a region for AWS in `~/.aws/config`

```ini
[default]
region=us-east-1
```

You can configure other settings in the [configuration file](plugin.cloud-inference.ini.example) at `/opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/nxai_manager/etc/plugin.cloud-inference.ini`:

```ini
[common]
debug_level=DEBUG
[inference]
image_path=/opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/nxai_manager/postprocessors/face.png
```

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

# Add Cloud Inference Postprocessor Python project
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/postprocessor-cloud-inference-example)

# # Add installation option
install(TARGETS
    DESTINATION /opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/nxai_manager/postprocessors/
)
install(PROGRAMS
    ${CMAKE_CURRENT_BINARY_DIR}/postprocessor-cloud-inference-example/postprocessor-cloud-inference-example
    DESTINATION /opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/nxai_manager/postprocessors/
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

A convenience directory within the AI Manager installation is created for this purpose at `/opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/nxai_manager/postprocessors`.

The application and settings files you add must be readable and executable by the NX AI AI Manager. This can be achieved by running:

```
sudo chmod -R 777 /opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/nxai_manager/postprocessors
```

Install the postprocessor automatically with the cmake command, also from within the *build* directory.

```shell
cmake --build . --target install
```

## Defining the postprocessor

Create a configuration file at
```
/opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/nxai_manager/postprocessors/external_postprocessors.json
```
for Linux, or
```
C:\Program Files\Network Optix\Nx Meta\MediaServer\plugins\nxai_plugin\nxai_manager\postprocessors\external_postprocessors.json
```
for Windows, and add the details of your postprocessor to the root object of that file. For example: 

``` json
{
    "externalPostprocessors": [
        {
            "Name":"Cloud-Inference-Postprocessor",
            "Command":"/opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/nxai_manager/postprocessors/postprocessor-cloud-inference-example",
            "SocketPath":"/tmp/python-cloud-inference-postprocessor.sock",
            "ReceiveInputTensor": true
        }
    ]
}
```
For Linux, and
```json
{
    "externalPostprocessors": [
        {
            "Name":"Cloud-Inference-Postprocessor",
            "Command":"C:\\Windows\\System32\\config\\systemprofile\\AppData\\Local\\Network Optix\\Network Optix MetaVMS Media Server\\nx_ai_manager\\nxai_manager\\postprocessors\\postprocessor-cloud-inference-example.exe",
            "SocketPath":"C:\\Users\\user\\AppData\\Local\\Temp\\example-cloud-postprocessor.sock",
            "ReceiveInputTensor": true,
        }
    ]
}
```
For Windows.

This tells the AI Manager about the postprocessor:
- **Name** gives the postprocesor a name so it can be selected later
- **Command** defines how to start the postprocessor
- **SocketPath** tells the AI Manager where to send data to so the external postprocessor will receive it
- **ReceiveInputTensor** tells the AI Manager if this postprocessor expects information to access the raw input tensor data

## Restarting the server

Finally, to (re)load your new postprocessor, make sure to restart the NX Server with:

```shell
sudo service networkoptix-metavms-mediaserver restart
```

You also want to make sure the postprocessor can be used by the NX AI Manager (this is the mostly same command as earlier)

```shell
sudo chmod -R a+x /opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/nxai_manager/postprocessors/
```

## Selecting to the postprocessor

If the postprocessor is defined correctly, its name should appear in the list of postprocessors in the NX Plugin settings. If it is selected in the plugin settings then the Edge AI Runtime will send data to the postprocessor and wait for its output.

## Output logging

There is an output log where the uploads can be tracked in real time from the server.

```shell
tail -f /opt/networkoptix-metavms/mediaserver/var/nx_ai_manager/nxai_manager/etc/plugin.cloud-inference.log
```

# Licence

Copyright 2025, Network Optix, All rights reserved.