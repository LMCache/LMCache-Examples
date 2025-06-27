#!/bin/bash

# Container names used in run-server.sh
CONTAINER_NAME_LMCACHE="kv-blending-server-lmcache"
CONTAINER_NAME_DEFAULT="kv-blending-server-default"

echo "Stopping and removing KV Blending Demo containers..."

# Stop containers
if [ "$(sudo docker ps -q -f name=${CONTAINER_NAME_LMCACHE})" ]; then
    echo "Stopping ${CONTAINER_NAME_LMCACHE}..."
    sudo docker stop "${CONTAINER_NAME_LMCACHE}"
else
    echo "${CONTAINER_NAME_LMCACHE} not found or not running."
fi

if [ "$(sudo docker ps -q -f name=${CONTAINER_NAME_DEFAULT})" ]; then
    echo "Stopping ${CONTAINER_NAME_DEFAULT}..."
    sudo docker stop "${CONTAINER_NAME_DEFAULT}"
else
    echo "${CONTAINER_NAME_DEFAULT} not found or not running."
fi

# Remove containers
# Add -f to docker ps -a to filter by name even if stopped
if [ "$(sudo docker ps -a -q -f name=${CONTAINER_NAME_LMCACHE})" ]; then
    echo "Removing ${CONTAINER_NAME_LMCACHE}..."
    sudo docker rm "${CONTAINER_NAME_LMCACHE}"
else
    echo "${CONTAINER_NAME_LMCACHE} (for removal) not found."
fi

if [ "$(sudo docker ps -a -q -f name=${CONTAINER_NAME_DEFAULT})" ]; then
    echo "Removing ${CONTAINER_NAME_DEFAULT}..."
    sudo docker rm "${CONTAINER_NAME_DEFAULT}"
else
    echo "${CONTAINER_NAME_DEFAULT} (for removal) not found."
fi

echo "Cleanup complete."
# Note: This script does not remove the Docker image itself (kv-blending-backend).
# To remove the image: docker rmi kv-blending-backend:latest
# Also, it doesn't clear LMCache's disk cache if one was configured and used inside the container volumes (not the case here yet).
# Or host-mounted cache directories (not used in current run-server.sh).
# The current LMCache setup in vllm_server.py uses LMCACHE_LOCAL_CPU=True, so cache is in memory or RAM-disk equivalent within container.
# If LMCACHE_LOCAL_DISK was enabled with a path inside the container, it would be removed with the container unless it was a mounted volume.
