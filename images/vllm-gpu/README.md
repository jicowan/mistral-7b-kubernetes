# vLLM GPU Image

High-performance Mistral 7B inference using vLLM with NVIDIA GPUs.

## Features

- **Base**: NVIDIA CUDA 12.9 with cuDNN
- **Framework**: vLLM 0.6+ with PyTorch 2.4+
- **Hardware**: NVIDIA A10G, L4, V100, A100
- **Performance**: Highest throughput with CUDA graphs
- **Memory**: Optimized GPU memory utilization

## Quick Start

```bash
# Build
./build.sh

# Run locally
docker run -p 8000:8000 --gpus all vllm-mistral-7b:latest

# Test
python test_client.py
```

## Configuration

### Environment Variables
- `MODEL_NAME`: Hugging Face model ID (default: mistralai/Mistral-7B-Instruct-v0.3)
- `MAX_MODEL_LEN`: Maximum sequence length (default: 32768)
- `GPU_MEMORY_UTILIZATION`: GPU memory usage (default: 0.9)
- `TENSOR_PARALLEL_SIZE`: Number of GPUs (default: 1)

### GPU Requirements
- **Minimum**: 16GB VRAM
- **Recommended**: 24GB VRAM (A10G, L4)
- **Compute Capability**: 7.0+

## Performance Tuning

### For A10G/L4 (24GB)
```yaml
env:
- name: GPU_MEMORY_UTILIZATION
  value: "0.9"
- name: MAX_MODEL_LEN
  value: "32768"
```

### For V100/T4 (16GB)
```yaml
env:
- name: GPU_MEMORY_UTILIZATION
  value: "0.8"
- name: MAX_MODEL_LEN
  value: "16384"
```

## Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f kubernetes-deployment.yaml

# Port forward
kubectl port-forward service/vllm-mistral-7b-service 8000:8000

# Test
python test_client.py
```

## API Endpoints

- `POST /generate` - Text generation
- `GET /health` - Health check
- `GET /models` - List models
- `GET /` - Server info

## Troubleshooting

### Out of Memory
- Reduce `GPU_MEMORY_UTILIZATION` to 0.8
- Reduce `MAX_MODEL_LEN` to 16384

### Slow Performance
- Ensure CUDA graphs are enabled (`enforce_eager=False`)
- Check GPU utilization with `nvidia-smi`
- Verify tensor parallel size matches GPU count
