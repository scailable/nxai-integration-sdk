#!/bin/bash

# Build the image and capture the ID directly
image_id=$(docker build -q .)
echo "Image ID: $image_id"
echo "Current Directory: ${PWD}"
mkdir -p build/postprocessors
mkdir -p build/preprocessors

# Create container with proper options
container_id=$(
    docker create \
        --mount type=bind,src=$PWD,dst=/app \
        $image_id
)

echo "Container ID: $container_id"

# Start container and show output
docker start -a $container_id

#docker stop $(docker ps -a -q) && docker rm $(docker ps -a -q)
docker stop $container_id
docker rm $container_id
