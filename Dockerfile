# importing images
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# changing user to root
USER root

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

# installing some necessary libraries
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get upgrade -y

# Python package management and basic dependencies
RUN apt-get install -y curl python3.10 python3.10-dev python3.10-distutils

# Register the version in alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Set python 3 as the default python
RUN update-alternatives --set python /usr/bin/python3.10

# Upgrade pip to latest version
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py

# install torch
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121