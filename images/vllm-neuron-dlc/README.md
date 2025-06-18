# vLLM + AWS Neuron DLC for Mistral 7B on Inferentia2

This is a **simplified, production-ready** approach using AWS's official Neuron Deep Learning Container with vLLM's built-in Neuron support.

## Why This Approach?

- ✅ **AWS Official**: Uses AWS's battle-tested Neuron DLC
- ✅ **vLLM Native**: Leverages vLLM's built-in Neuron support
- ✅ **Simple**: No custom compilation logic needed
- ✅ **Reliable**: Proven to work with Mistral models on Inf2
- ✅ **OpenAI Compatible**: Drop-in replacement for OpenAI API

## Quick Start

### Build and Push
```bash
cd images/vllm-neuron-dlc
./build.sh

# Push to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 820537372947.dkr.ecr.us-west-2.amazonaws.com
docker push 820537372947.dkr.ecr.us-west-2.amazonaws.com/vllm-mistral-7b-neuron:latest
```

### Deploy to Kubernetes
```bash
kubectl apply -f kubernetes-deployment.yaml
```

### Test the API
```bash
# Wait for pod to be ready
kubectl wait --for=condition=ready pod -l app=vllm-mistral-7b-neuron --timeout=300s

# Port forward
kubectl port-forward svc/vllm-mistral-7b-neuron 8000:8000

# Test OpenAI-compatible API
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "prompt": "What is machine learning?",
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

## How It Works

1. **AWS Neuron DLC**: Provides optimized PyTorch + Neuron runtime
2. **vLLM**: Handles model compilation and serving automatically
3. **Automatic Inf2 Targeting**: vLLM detects Inferentia2 and compiles accordingly
4. **OpenAI API**: Standard API interface for easy integration

## Configuration

Environment variables in the deployment:
- `MODEL_ID`: Hugging Face model identifier
- `DEVICE`: Set to "neuron" for Inferentia2
- `TENSOR_PARALLEL_SIZE`: Number of Neuron cores (2 for inf2.xlarge)
- `MAX_NUM_SEQS`: Maximum concurrent sequences

## Advantages Over Custom Implementation

- **No custom compilation logic** - vLLM handles it
- **No lifespan function complexity** - vLLM manages model lifecycle  
- **No manual target configuration** - vLLM auto-detects Inf2
- **Production-ready** - Used by many companies in production
- **Better error handling** - vLLM has robust error handling
- **Automatic optimization** - vLLM applies best practices automatically
