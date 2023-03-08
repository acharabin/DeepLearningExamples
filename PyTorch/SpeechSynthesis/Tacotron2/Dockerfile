ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:22.10-py3
FROM ${FROM_IMAGE_NAME}

ADD . /workspace/tacotron2
WORKDIR /workspace/tacotron2
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install boto3
RUN pip install GPUtil
RUN pip install configparser
RUN pip install omegaconf

ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4
