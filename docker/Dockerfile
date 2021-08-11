FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

MAINTAINER Ryuichi YAMAMOTO <zryuichi@gmail.com>

RUN apt-get update && apt-get install -y curl git libsamplerate0 libsndfile1 && apt-get clean

RUN pip install ttslearn
