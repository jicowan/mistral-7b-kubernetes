# AWS Deep Learning Container for Neuron + Mistral 7B
# Using AWS PyTorch Neuron DLC - optimized for Inferentia
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuronx:2.1.2-neuronx-py310-sdk2.18.2-ubuntu20.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV NEURON_RT_NUM_CORES=2
ENV NEURON_CC_FLAGS="--model-type=transformer"

# AWS Neuron DLC already includes:
# - PyTorch 2.1.2 with Neuron extensions
# - Python 3.10
# - AWS Neuron SDK 2.18.2
# - torch-neuronx
# - neuronx-distributed
# - Optimized Neuron runtime
# - AWS-specific performance tunings

# Install additional system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip (ensure latest version)
RUN pip install --upgrade pip

# Install web server dependencies (not included in Neuron DLC)
RUN pip install \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    requests==2.31.0

# Install additional ML dependencies with updated versions for tokenizer compatibility
RUN pip install --upgrade \
    transformers>=4.35.0 \
    tokenizers>=0.15.0 \
    accelerate==0.24.1 \
    sentencepiece==0.1.99

# Create app directory
WORKDIR /app

# Create directory for compiled models
RUN mkdir -p /tmp/neuron_compiled_model

# Copy application files
COPY neuron_server.py .
COPY neuron_compile.py .
COPY requirements.txt .

# Expose port
EXPOSE 8000

# Extended health check for Neuron (compilation takes time)
HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# AWS Neuron DLC optimizations (already configured in base image)
# These are set by default in the DLC:
# ENV NEURON_RT_LOG_LEVEL=INFO
# ENV NEURON_FRAMEWORK=PYTORCH
# ENV NEURON_RT_VISIBLE_CORES=0,1

# Additional Neuron optimizations
ENV NEURON_RT_EXEC_TIMEOUT=60
ENV NEURON_RT_LOAD_TIMEOUT=60

# Run the application
CMD ["python", "neuron_server.py"]
