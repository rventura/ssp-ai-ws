#!/bin/sh

# Configure here an available port
PORT=8888

# Docker image
IMAGE=quay.io/jupyter/pytorch-notebook
CHOME=/home/jovyan

# Run the docker
docker run -t --rm -v ./ssp:${CHOME}/ssp -p ${PORT}:8888 ${IMAGE}

# EOF
