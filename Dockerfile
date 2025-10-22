# Use Ubuntu 22.04 as the base image
FROM ubuntu:22.04

# Set working directory to /app
WORKDIR /app

# Set non-interactive frontend for package installations
ENV DEBIAN_FRONTEND=noninteractive

# Install required dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-venv \
    python3-dev \
    patchelf \
    ccache \
    libhdf5-dev \
    tzdata \
    cargo \
    wget \
    gpg \
    ca-certificates \
    g++ \
    lsb-release && \
    rm -rf /var/lib/apt/lists/*

# Add the Kitware repository, update, and install the newer cmake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null && \
    echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/kitware.list >/dev/null && \
    apt-get update && \
    apt-get install -y cmake && \
    rm -rf /var/lib/apt/lists/*

# Build all the processors and copy them to build folder
CMD bash /app/build_all.sh
