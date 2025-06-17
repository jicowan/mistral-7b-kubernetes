# Container Images Directory

This directory contains all container images for the Mistral 7B inference server, organized by deployment type.

## Directory Structure

```
images/
├── vllm-gpu/              # vLLM + NVIDIA GPUs (Standard)
├── triton-gpu/            # Triton + vLLM + NVIDIA GPUs
├── neuron-inferentia/     # AWS Neuron + Inferentia chips
├── vllm-dlc/             # vLLM + AWS Deep Learning Containers
├── triton-dlc/           # Triton + AWS Deep Learning Containers
└── neuron-dlc/           # Neuron + AWS Deep Learning Containers
```

## Each Directory Contains

- `Dockerfile` - Container definition
- `build.sh` - Individual build script
- `requirements.txt` - Python dependencies
- `kubernetes-deployment.yaml` - Kubernetes deployment
- Application files (servers, clients, etc.)

## Building Images

### Build All Images
```bash
# From project root
./build-all-images.sh all
```

### Build Specific Image
```bash
# From project root
./build-all-images.sh vllm-gpu

# Or from image directory
cd images/vllm-gpu
./build.sh
```

### Build and Push to ECR
```bash
# From project root
./build-all-images.sh vllm-gpu us-west-2 123456789012

# Or from image directory
cd images/vllm-gpu
./build.sh latest 123456789012.dkr.ecr.us-west-2.amazonaws.com
```

## Image Comparison

| Image | Base | Hardware | Performance | Cost | Use Case |
|-------|------|----------|-------------|------|----------|
| **vllm-gpu** | CUDA 12.9 | NVIDIA GPU | Highest | High | Development, High Performance |
| **triton-gpu** | NVIDIA Triton | NVIDIA GPU | High | High | Production, Model Management |
| **neuron-inferentia** | Neuron SDK | AWS Inferentia | Medium | Low | Cost-Optimized Production |
| **vllm-dlc** | AWS DLC | NVIDIA GPU | Highest | High | AWS-Optimized Performance |
| **triton-dlc** | AWS DLC | NVIDIA GPU | High | High | AWS-Optimized Production |
| **neuron-dlc** | AWS DLC | AWS Inferentia | Medium | Low | AWS-Optimized Cost |

## Quick Start Examples

### vLLM GPU (Recommended for Development)
```bash
cd images/vllm-gpu
./build.sh
docker run -p 8000:8000 --gpus all vllm-mistral-7b:latest
```

### Triton DLC (Recommended for Production)
```bash
cd images/triton-dlc
./build.sh
docker run -p 8000:8000 --gpus all triton-mistral-7b-dlc:latest
```

### Neuron Inferentia (Recommended for Cost)
```bash
cd images/neuron-inferentia
./build.sh
docker run -p 8000:8000 neuron-mistral-7b:latest
```

## Testing

Each image directory includes test clients:
- `test_client.py` - Standard test client
- `triton_test_client.py` - Triton-specific client
- `neuron_test_client.py` - Neuron-specific client

## Deployment

Each directory includes Kubernetes deployment files optimized for the specific platform:
- Resource requests/limits
- Node selectors
- Health checks
- Environment variables
