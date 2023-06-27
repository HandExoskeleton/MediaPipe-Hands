#!/bin/bash

# Build Docker image
docker build -t mediapipe-hands .

# Run Docker container
docker run --privileged --device=/path/to/webcam_device -it mediapipe-hands
