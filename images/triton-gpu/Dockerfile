FROM nvcr.io/nvidia/tritonserver:24.01-py3

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install vLLM
RUN pip install vllm

# Install additional dependencies
RUN pip install \
    transformers \
    accelerate \
    sentencepiece

# Create model repository directory
RUN mkdir -p /models

# Copy model repository
COPY triton-model-repository/ /models/

# Set model repository path
ENV TRITON_MODEL_REPOSITORY=/models

# Expose Triton ports
EXPOSE 8000 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/v2/health/ready || exit 1

# Start Triton server
CMD ["tritonserver", "--model-repository=/models", "--allow-gpu-metrics=true", "--allow-cpu-metrics=true"]
