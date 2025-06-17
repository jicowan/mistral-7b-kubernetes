FROM nvidia/cuda:12.9.0-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.4 support (closest to CUDA 12.9)
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install vLLM (latest version compatible with newer CUDA)
RUN pip install vllm>=0.6.0

# Install additional dependencies with updated versions
RUN pip install \
    fastapi>=0.110.0 \
    uvicorn[standard]>=0.27.0 \
    pydantic>=2.6.0 \
    transformers>=4.40.0 \
    accelerate>=0.28.0 \
    sentencepiece>=0.2.0

# Create app directory
WORKDIR /app

# Copy application files
COPY vllm_server.py .
COPY requirements.txt .

# Set additional environment variables for CUDA 12.9 optimization
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
ENV NCCL_DEBUG=INFO

# Expose port
EXPOSE 8000

# Health check with longer startup time for newer versions
HEALTHCHECK --interval=30s --timeout=30s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "vllm_server.py"]
