# Triton-Compatible Server with AWS DLC

This directory contains a complete Triton-compatible implementation using AWS Deep Learning Containers as the base, with vLLM as the inference backend.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Triton-Compatible API                    │
├─────────────────────────────────────────────────────────────┤
│  /v2/health/ready  │  /v2/models  │  /v2/models/{}/infer   │
├─────────────────────────────────────────────────────────────┤
│                    FastAPI Server                          │
├─────────────────────────────────────────────────────────────┤
│                    vLLM Engine                             │
├─────────────────────────────────────────────────────────────┤
│              AWS PyTorch Training DLC                      │
├─────────────────────────────────────────────────────────────┤
│                 NVIDIA GPU / CUDA                          │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
cd aws-dlc
./build-and-deploy-triton.sh
```

## Features

### Triton-Compatible Endpoints
- `GET /v2/health/ready` - Readiness check
- `GET /v2/health/live` - Liveness check  
- `GET /v2/models` - List models
- `GET /v2/models/{model}` - Model metadata
- `POST /v2/models/{model}/infer` - Inference
- `GET /v2` - Server metadata

### AWS DLC Benefits
- Pre-optimized PyTorch 2.1.0 + CUDA 12.1
- 50-70% faster build times
- AWS-specific performance tunings
- Regular security updates

### Performance Features
- vLLM backend for high throughput
- CUDA graphs for GPU efficiency
- Dynamic batching capability
- Memory optimization

## Testing

```bash
# Port forward
kubectl port-forward service/triton-mistral-7b-dlc-service 8000:8000

# Run tests
python triton_test_client.py
```

## Monitoring

```bash
# Check logs
kubectl logs -l app=triton-mistral-7b-dlc -f

# Scale deployment
kubectl scale deployment triton-mistral-7b-dlc --replicas=2

# Check metrics
kubectl port-forward service/triton-mistral-7b-dlc-service 8002:8002
```
