#!/bin/bash
#
# Docker Entrypoint for DataSentience SageMaker Container
# Configures FastAPI application for production deployment
#
# SageMaker calls this script with "serve" argument for inference
# Training mode is not implemented for this inference-only container
#

set -e

if [ "$1" = "serve" ]; then
    echo "Starting SageMaker inference server..."
    # Production configuration:
    # - 4 worker processes for concurrent request handling
    # - uvicorn ASGI workers for async FastAPI support
    # - Port 8080 required by SageMaker
    # - 120 second timeout for NVIDIA NIM API calls
    exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8080 --timeout 120 src.main:app
elif [ "$1" = "train" ]; then
    echo "Training not implemented for inference container"
    exit 1
else
    # Allow custom commands for debugging
    echo "Running command: $@"
    exec "$@"
fi