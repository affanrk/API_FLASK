#!/bin/bash

# Install SDL2 dependencies
apt-get update && apt-get install -y \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libportmidi-dev \
    libswscale-dev \
    libsmpeg-dev \
    libavformat-dev \
    libavcodec-dev \
    libfreetype6-dev \
    libx11-6

# Install Python dependencies
pip install -r requirements.txt
