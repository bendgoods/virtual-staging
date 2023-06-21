# Pull a base image.
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Install libraries in the brand new image. 
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         python3-pip \
         python3-setuptools \
         python3-dev \
         build-essential \
         nginx \
         git \
         ca-certificates \
         libglib2.0-0 \
         libsm6 \
         libxrender1 \
         libxext6 \
         zlib1g-dev \
         libjpeg-dev \
         libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Python 3 setup
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

# Set variables
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# Here we get all python packages.
# There's substantial overlap between scipy and numpy that we eliminate by
# linking them together. Likewise, pip leaves the install caches populated which uses
# a significant amount of space. These optimizations save a fair amount of space in the
# image, which reduces start up time.
RUN pip --no-cache-dir install pandas awscli==1.16.253 fastapi uvicorn gunicorn boto3==1.9.243 botocore==1.12.243

# Set an executable path
ENV PATH="/opt/program:${PATH}"


# Set the working directory for all the subsequent Dockerfile instructions.

ADD requirements.txt requirements.txt
RUN pip --no-cache-dir  install -r requirements.txt
WORKDIR /opt/program
COPY weights weights 

COPY models models
COPY settings settings
COPY pipelines pipelines    
COPY load_model.py load_model.py
COPY utils utils
RUN pip install boto3 --upgrade

RUN python load_model.py

COPY . .
COPY sagemaker_files .


RUN chmod +x /opt/program/serve
