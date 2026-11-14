# hash:sha256:0000000000000000000000000000000000000000000000000000000000000000
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/matlab:2023b-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG MLM_LICENSE_FILE

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --no-cache-dir \
    h5py==3.11.0 \
    matplotlib==3.8.4 \
    numpy==1.26.4 \
    scipy==1.13.1

# PyTorch for training (run --mode train). The Linux wheel bundles the CUDA
# 12.1 runtime, so it uses the GPU of a GPU machine (only the host driver is
# needed) and runs on the CPU elsewhere.
RUN pip install --no-cache-dir torch==2.4.1

COPY postInstall /
RUN /postInstall
