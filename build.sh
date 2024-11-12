#!/bin/bash

# Install necessary system dependencies
apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

# Install Python dependencies
pip install -r requirements.txt
