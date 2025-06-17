FROM public.ecr.aws/neuron/pytorch-inference-neuronx:2.1.2-neuronx-py310-sdk2.18.2-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV NEURON_RT_NUM_CORES=2
ENV NEURON_CC_FLAGS="--model-type=transformer"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip

# Install required packages
RUN pip install \
    fastapi \
    uvicorn \
    pydantic \
    transformers \
    accelerate \
    sentencepiece \
    torch-neuronx \
    neuronx-distributed

# Create app directory
WORKDIR /app

# Create directory for compiled models
RUN mkdir -p /tmp/neuron_compiled_model

# Copy application files
COPY neuron_server.py .
COPY requirements.txt .

# Install additional requirements
RUN pip install -r requirements.txt

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "neuron_server.py"]
