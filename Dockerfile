# Use Ubuntu 20.04 as the base image
FROM ubuntu:20.04

# Set working directory to /app
WORKDIR /app

# Install required dependencies
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt install -y python3-pip patchelf ccache libhdf5-dev tzdata cmake cargo && \
    rm -rf /var/lib/apt/lists/*

# Create plugin directories
RUN mkdir -p /opt/networkoptix-metavms/mediaserver/bin/plugins/nx_ai_manager_plugin/nxai_manager/postprocessors/
RUN mkdir -p /opt/networkoptix-metavms/mediaserver/bin/plugins/nx_ai_manager_plugin/nxai_manager/preprocessors/

# Build all the processors and copy them to build folder
CMD bash /app/build_all.sh
