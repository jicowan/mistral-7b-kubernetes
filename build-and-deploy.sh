#!/bin/bash

# Build and Deploy vLLM Mistral 7B to Kubernetes
# Optimized for NVIDIA A10G and L4 GPUs

set -e

# Configuration
IMAGE_NAME="vllm-mistral-7b"
IMAGE_TAG="latest"
REGISTRY="" # Set your container registry here (e.g., your-registry.com/)

echo "🚀 Building and Deploying vLLM Mistral 7B Server"
echo "================================================"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "1. Checking prerequisites..."
if ! command_exists docker; then
    echo "❌ Docker is not installed"
    exit 1
fi

if ! command_exists kubectl; then
    echo "❌ kubectl is not installed"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Build Docker image
echo ""
echo "2. Building Docker image..."
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Docker build failed"
    exit 1
fi

# Tag and push to registry (if registry is set)
if [ ! -z "$REGISTRY" ]; then
    echo ""
    echo "3. Pushing to registry..."
    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY}${IMAGE_NAME}:${IMAGE_TAG}
    docker push ${REGISTRY}${IMAGE_NAME}:${IMAGE_TAG}
    
    if [ $? -eq 0 ]; then
        echo "✅ Image pushed to registry"
        # Update deployment YAML with registry
        sed -i.bak "s|image: ${IMAGE_NAME}:${IMAGE_TAG}|image: ${REGISTRY}${IMAGE_NAME}:${IMAGE_TAG}|g" kubernetes-deployment.yaml
    else
        echo "❌ Failed to push to registry"
        exit 1
    fi
else
    echo ""
    echo "3. Skipping registry push (REGISTRY not set)"
    echo "   Make sure your Kubernetes cluster can access the local image"
fi

# Check if NVIDIA GPU operator is installed
echo ""
echo "4. Checking GPU support..."
kubectl get nodes -o json | jq -r '.items[].status.allocatable | keys[]' | grep -q "nvidia.com/gpu"
if [ $? -eq 0 ]; then
    echo "✅ NVIDIA GPU support detected"
else
    echo "⚠️  NVIDIA GPU support not detected. Make sure NVIDIA GPU Operator is installed."
    echo "   Install with: kubectl apply -f https://raw.githubusercontent.com/NVIDIA/gpu-operator/main/deployments/gpu-operator/values.yaml"
fi

# Deploy to Kubernetes
echo ""
echo "5. Deploying to Kubernetes..."
kubectl apply -f kubernetes-deployment.yaml

if [ $? -eq 0 ]; then
    echo "✅ Deployment applied successfully"
else
    echo "❌ Deployment failed"
    exit 1
fi

# Wait for deployment to be ready
echo ""
echo "6. Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=600s deployment/vllm-mistral-7b

if [ $? -eq 0 ]; then
    echo "✅ Deployment is ready"
else
    echo "❌ Deployment failed to become ready within timeout"
    echo "Check logs with: kubectl logs -l app=vllm-mistral-7b"
    exit 1
fi

# Get service information
echo ""
echo "7. Service Information:"
kubectl get service vllm-mistral-7b-service

# Show pod status
echo ""
echo "8. Pod Status:"
kubectl get pods -l app=vllm-mistral-7b

# Show logs
echo ""
echo "9. Recent Logs:"
kubectl logs -l app=vllm-mistral-7b --tail=20

echo ""
echo "🎉 Deployment completed successfully!"
echo ""
echo "To test the deployment:"
echo "1. Port forward: kubectl port-forward service/vllm-mistral-7b-service 8000:8000"
echo "2. Run test client: python test_client.py"
echo ""
echo "To check logs: kubectl logs -l app=vllm-mistral-7b -f"
echo "To scale: kubectl scale deployment vllm-mistral-7b --replicas=2"
