# Base image with Python 3.12.3
FROM python:3.12.3-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    yasm \
    nasm \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    libv4l-dev \
    libx264-dev \
    libx265-dev \
    libnuma-dev \
    libvpx-dev \
    libmp3lame-dev \
    libopus-dev \
    libass-dev \
    libfreetype-dev \
    ca-certificates \
    wget \
    unzip \
    gnupg \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libopenblas-dev && \
    # Add NVIDIA repository for CUDA with modern keyring
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/3bf863cc.pub | gpg --dearmor -o /etc/apt/keyrings/nvidia-cuda.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nvidia-cuda.gpg] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    cuda-libraries-12-8 \
    cuda-libraries-dev-12-8 && \
    rm -rf /var/lib/apt/lists/*

# Install Go 1.24.3 (to match go.mod)
RUN wget https://go.dev/dl/go1.24.3.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.24.3.linux-amd64.tar.gz && \
    rm go1.24.3.linux-amd64.tar.gz && \
    ln -sf /usr/local/go/bin/go /usr/bin/go

# === Build FFmpeg with CUDA/NVENC support ===
RUN mkdir -p /tmp/ffmpeg && cd /tmp/ffmpeg && \
    git clone --depth=1 https://git.ffmpeg.org/ffmpeg.git ffmpeg && \
    git clone --depth=1 https://git.videolan.org/git/ffmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && make install PREFIX=/usr/local && \
    cd ../ffmpeg && \
    ./configure \
    --prefix=/usr/local \
    --enable-cuda \
    --enable-cuvid \
    --enable-nvenc \
    --enable-nonfree \
    --enable-gpl \
    --enable-libx264 \
    --enable-libx265 \
    --enable-libmp3lame \
    --enable-libopus \
    --enable-libvpx \
    --enable-libass \
    --enable-libfreetype \
    --disable-debug \
    --disable-doc \
    --extra-cflags=-I/usr/local/cuda/include \
    --extra-ldflags=-L/usr/local/cuda/lib64 \
    --enable-ffnvcodec && \
    make -j"$(nproc)" && \
    make install && \
    hash -r && \
    ffmpeg -hwaccels && \
    cd / && rm -rf /tmp/ffmpeg

# Install PyTorch 2.7.1 with CUDA 12.8
RUN pip install --no-cache-dir torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Go env
ENV GOPATH=/go
ENV PATH=$GOPATH/bin:/usr/local/go/bin:/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Copy and build Go binary
COPY go/ ./go/
WORKDIR /app/go
RUN go mod tidy && \
    go build -o /app/extract.exe ./cmd/frame_extractor/main.go
WORKDIR /app

# Copy Python dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python source and other files
COPY src/ ./src/
COPY .env .env
COPY database.db database.db

EXPOSE 8000

# Default command
CMD ["python3", "src/run.py"]