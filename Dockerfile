# Use a base image with Python
FROM python:3.9

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the code into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir opencv-python mediapipe

# Set the default command to run when the container starts
CMD ["python", "main.py"]
