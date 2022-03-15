FROM pytorch/pytorch:1.9.1-cuda11.1-cudnn8-runtime

RUN apt-get update && apt-get install -y \
  vim \
  git \
  make

WORKDIR /gpt2-compression

COPY . .

RUN pip install --no-cache-dir -U -r requirements.txt -e .
