# WhisperX ASR API Service Dockerfile
#
# Build args (override per image variant):
#   TORCH_VERSION   - PyTorch version to install (default 2.11.0, broadly
#                     compatible from Pascal through Hopper; bumped from 2.7.1
#                     for CVE fixes in the cu126 line, see ci-security checks).
#   TORCH_INDEX_URL - PyTorch wheel index URL (default cu126). For Blackwell
#                     (RTX 50xx) use TORCH_VERSION=2.8.0 with cu128.
#
# Image structure notes:
# - ubuntu:22.04 base instead of nvidia/cuda devel: the torch and
#   ctranslate2 wheels bundle the CUDA runtime they need, and the NVIDIA
#   container runtime injects the driver libraries. The devel toolchain
#   (~8 GB) was never used at runtime.
# - All pip installs in a single layer so the WhisperX-induced torch
#   upgrade and subsequent re-pin do not persist as dead layers
#   (~10 GB of stale torch copies in the previous layering).
ARG TORCH_VERSION=2.11.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

FROM ubuntu:22.04

ARG TORCH_VERSION
ARG TORCH_INDEX_URL

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    ffmpeg \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Single layer: torch pin -> WhisperX (silently upgrades torch) -> pyannote
# -> torch re-pin -> transformers -> API deps. Intermediate states never
# materialize as image layers.
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
        torch==${TORCH_VERSION} \
        torchaudio==${TORCH_VERSION} \
        --index-url ${TORCH_INDEX_URL} && \
    pip3 install --no-cache-dir git+https://github.com/sealambda/whisperX.git@feat/pyannote-audio-4 && \
    sed -i 's/use_token=/token=/g' \
        /usr/local/lib/python3.10/dist-packages/whisperx/diarize.py && \
    pip3 install --no-cache-dir --upgrade pyannote.audio && \
    pip3 install --no-cache-dir \
        torch==${TORCH_VERSION} \
        torchaudio==${TORCH_VERSION} \
        --index-url ${TORCH_INDEX_URL} && \
    pip3 install --no-cache-dir "transformers>=5.13,<6" && \
    pip3 install --no-cache-dir \
        fastapi==0.141.1 \
        "uvicorn[standard]==0.52.4" \
        python-multipart==0.0.32 \
        pydantic==2.13.5 \
        prometheus-client==0.20.0 \
        "ray[serve]>=2.9" \
        "protobuf<7" \
        "setuptools>=83"

# Prefer torch's bundled cuDNN over any system cuDNN
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

# Pre-download NLTK data for timestamp alignment (enables offline use)
RUN python3 -c "import nltk; nltk.download('punkt_tab', download_dir='/.cache/nltk_data')"
ENV NLTK_DATA=/.cache/nltk_data

RUN mkdir -p /.cache && chmod 777 /.cache
ENV HF_HOME=/.cache

COPY app /workspace/app
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

EXPOSE 9000 8265

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:9000/health')" || exit 1

ENV SERVE_MODE=simple

CMD ["/workspace/entrypoint.sh"]
