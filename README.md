# Mistral 7B Server for Kubernetes

High-performance inference server for Mistral 7B Instruct with multiple deployment options:
- **vLLM + NVIDIA GPUs** (A10G, L4) - Highest performance
- **Triton + vLLM + NVIDIA GPUs** - Production features with dynamic batching
- **AWS Neuron + Inferentia** (Inf1, Inf2) - Cost-effective AWS-native inference

## Features

- **High Performance**: Uses vLLM for fast inference with CUDA graphs
- **GPU Optimized**: Configured for NVIDIA A10G (24GB) and L4 (24GB) GPUs
- **Production Ready**: Includes health checks, monitoring, and auto-scaling
- **Kubernetes Native**: Complete deployment manifests included
- **RESTful API**: FastAPI-based server with OpenAPI documentation

## Quick Start

Choose your deployment option:

### Option 1: vLLM + NVIDIA GPUs (Recommended for Performance)
```bash
./build-and-deploy.sh
kubectl port-forward service/vllm-mistral-7b-service 8000:8000
python test_client.py
```

### Option 2: Triton + vLLM + NVIDIA GPUs (Production Features)
```bash
docker build -f Dockerfile.triton -t triton-vllm-mistral-7b:latest .
kubectl apply -f kubernetes-deployment-triton.yaml
kubectl port-forward service/triton-vllm-mistral-7b-service 8000:8000
python triton_client.py
```

### Option 3: AWS Neuron + Inferentia (Cost-Effective)
```bash
./build-and-deploy-neuron.sh
kubectl port-forward service/neuron-mistral-7b-service 8000:8000
python neuron_test_client.py
```

## GPU Requirements

| GPU Model | VRAM | Status | Notes |
|-----------|------|--------|-------|
| A10G | 24GB | ✅ Recommended | Excellent performance |
| L4 | 24GB | ✅ Recommended | Good price/performance |
| V100 | 16GB | ⚠️ Limited | May need reduced context |
| T4 | 16GB | ⚠️ Limited | May need reduced context |

## Inferentia Requirements

| Instance Type | Neuron Cores | Memory | Status | Notes |
|---------------|--------------|--------|--------|-------|
| inf1.xlarge | 4 | 8GB | ✅ Supported | Entry-level |
| inf1.2xlarge | 4 | 16GB | ✅ Recommended | Good balance |
| inf1.6xlarge | 16 | 48GB | ✅ High throughput | Multi-model |
| inf2.xlarge | 2 | 32GB | ✅ Latest gen | Best efficiency |
| inf2.8xlarge | 2 | 128GB | ✅ High memory | Large contexts |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_NAME` | `mistralai/Mistral-7B-Instruct-v0.3` | Hugging Face model ID |
| `MAX_MODEL_LEN` | `32768` | Maximum sequence length |
| `GPU_MEMORY_UTILIZATION` | `0.9` | GPU memory utilization (0.0-1.0) |
| `TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs for tensor parallelism |
| `TRUST_REMOTE_CODE` | `true` | Trust remote code in model |

### Resource Requests/Limits

- **GPU**: 1x NVIDIA GPU (A10G/L4 recommended)
- **Memory**: 16-24GB RAM
- **CPU**: 4-8 cores
- **Storage**: ~15GB for model weights

## API Endpoints

### Generate Text
```bash
POST /generate
{
  "prompt": "What is machine learning?",
  "max_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9
}
```

### Health Check
```bash
GET /health
```

### List Models
```bash
GET /models
```

## Monitoring and Scaling

### Check Pod Status
```bash
kubectl get pods -l app=vllm-mistral-7b
```

### View Logs
```bash
kubectl logs -l app=vllm-mistral-7b -f
```

### Scale Deployment
```bash
kubectl scale deployment vllm-mistral-7b --replicas=3
```

## Performance Tuning

### For A10G/L4 GPUs:
- `GPU_MEMORY_UTILIZATION=0.9` (use 90% of VRAM)
- `MAX_MODEL_LEN=32768` (full context length)
- Enable CUDA graphs for better throughput

### For Multiple GPUs:
- Set `TENSOR_PARALLEL_SIZE=2` for 2 GPUs
- Adjust resource requests accordingly

### Memory Optimization:
- Reduce `MAX_MODEL_LEN` if running out of memory
- Lower `GPU_MEMORY_UTILIZATION` to 0.8 for stability

## Troubleshooting

### Common Issues:

1. **Out of Memory**: Reduce `GPU_MEMORY_UTILIZATION` or `MAX_MODEL_LEN`
2. **Slow Startup**: Model download takes time on first run
3. **GPU Not Found**: Ensure NVIDIA GPU Operator is installed

### Debug Commands:
```bash
# Check GPU availability
kubectl describe nodes | grep nvidia.com/gpu

# Check pod events
kubectl describe pod -l app=vllm-mistral-7b

# Get detailed logs
kubectl logs -l app=vllm-mistral-7b --previous
```
