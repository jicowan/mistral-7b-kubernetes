#!/bin/bash

# Build and Deploy using AWS Deep Learning Containers
# Supports vLLM (GPU) and Neuron (Inferentia) deployments

set -e

# Configuration
DEPLOYMENT_TYPE=${1:-"vllm"}  # vllm or neuron
AWS_REGION=${AWS_REGION:-"us-west-2"}
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-"763104351884"}  # AWS DLC account
REGISTRY="" # Set your container registry here

echo "🚀 Building and Deploying with AWS Deep Learning Containers"
echo "=========================================================="
echo "Deployment Type: $DEPLOYMENT_TYPE"
echo "AWS Region: $AWS_REGION"

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

if ! command_exists aws; then
    echo "❌ AWS CLI is not installed"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Authenticate with ECR for AWS DLC
echo ""
echo "2. Authenticating with AWS ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

if [ $? -eq 0 ]; then
    echo "✅ ECR authentication successful"
else
    echo "❌ ECR authentication failed"
    exit 1
fi

# Build appropriate image based on deployment type
echo ""
echo "3. Building Docker image..."

if [ "$DEPLOYMENT_TYPE" = "vllm" ]; then
    IMAGE_NAME="vllm-mistral-7b-dlc"
    DOCKERFILE="Dockerfile.vllm-dlc"
    DEPLOYMENT_YAML="kubernetes-deployment-vllm-dlc.yaml"
    
    echo "Building vLLM image with AWS DLC..."
    docker build -f $DOCKERFILE -t ${IMAGE_NAME}:latest ../
    
elif [ "$DEPLOYMENT_TYPE" = "neuron" ]; then
    IMAGE_NAME="neuron-mistral-7b-dlc"
    DOCKERFILE="Dockerfile.neuron-dlc"
    DEPLOYMENT_YAML="kubernetes-deployment-neuron-dlc.yaml"
    
    echo "Building Neuron image with AWS DLC..."
    docker build -f $DOCKERFILE -t ${IMAGE_NAME}:latest ../
    
else
    echo "❌ Invalid deployment type. Use 'vllm' or 'neuron'"
    exit 1
fi

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Docker build failed"
    exit 1
fi

# Tag and push to registry (if registry is set)
if [ ! -z "$REGISTRY" ]; then
    echo ""
    echo "4. Pushing to registry..."
    docker tag ${IMAGE_NAME}:latest ${REGISTRY}${IMAGE_NAME}:latest
    docker push ${REGISTRY}${IMAGE_NAME}:latest
    
    if [ $? -eq 0 ]; then
        echo "✅ Image pushed to registry"
        # Update deployment YAML with registry
        sed -i.bak "s|image: ${IMAGE_NAME}:latest|image: ${REGISTRY}${IMAGE_NAME}:latest|g" $DEPLOYMENT_YAML
    else
        echo "❌ Failed to push to registry"
        exit 1
    fi
else
    echo ""
    echo "4. Skipping registry push (REGISTRY not set)"
fi

# Check device support
echo ""
echo "5. Checking device support..."

if [ "$DEPLOYMENT_TYPE" = "vllm" ]; then
    # Check GPU support
    kubectl get nodes -o json | jq -r '.items[].status.allocatable | keys[]' | grep -q "nvidia.com/gpu"
    if [ $? -eq 0 ]; then
        echo "✅ NVIDIA GPU support detected"
    else
        echo "⚠️  NVIDIA GPU support not detected. Make sure NVIDIA GPU Operator is installed."
    fi
    
elif [ "$DEPLOYMENT_TYPE" = "neuron" ]; then
    # Check Neuron support
    kubectl get daemonset -n kube-system | grep -q neuron-device-plugin
    if [ $? -eq 0 ]; then
        echo "✅ Neuron device plugin detected"
    else
        echo "⚠️  Neuron device plugin not detected. Installing..."
        kubectl apply -f https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/master/src/k8/k8s-neuron-device-plugin.yml
        kubectl rollout status daemonset/neuron-device-plugin -n kube-system --timeout=300s
    fi
fi

# Deploy to Kubernetes
echo ""
echo "6. Deploying to Kubernetes..."
kubectl apply -f $DEPLOYMENT_YAML

if [ $? -eq 0 ]; then
    echo "✅ Deployment applied successfully"
else
    echo "❌ Deployment failed"
    exit 1
fi

# Wait for deployment to be ready
echo ""
echo "7. Waiting for deployment to be ready..."

if [ "$DEPLOYMENT_TYPE" = "vllm" ]; then
    DEPLOYMENT_NAME="vllm-mistral-7b-dlc"
    SERVICE_NAME="vllm-mistral-7b-dlc-service"
    TIMEOUT="600s"
elif [ "$DEPLOYMENT_TYPE" = "neuron" ]; then
    DEPLOYMENT_NAME="neuron-mistral-7b-dlc"
    SERVICE_NAME="neuron-mistral-7b-dlc-service"
    TIMEOUT="1200s"  # Longer for Neuron compilation
fi

kubectl wait --for=condition=available --timeout=$TIMEOUT deployment/$DEPLOYMENT_NAME

if [ $? -eq 0 ]; then
    echo "✅ Deployment is ready"
else
    echo "❌ Deployment failed to become ready within timeout"
    echo "Check logs with: kubectl logs -l app=$DEPLOYMENT_NAME"
    exit 1
fi

# Get service information
echo ""
echo "8. Service Information:"
kubectl get service $SERVICE_NAME

# Show pod status
echo ""
echo "9. Pod Status:"
kubectl get pods -l app=$DEPLOYMENT_NAME

# Show logs
echo ""
echo "10. Recent Logs:"
kubectl logs -l app=$DEPLOYMENT_NAME --tail=20

echo ""
echo "🎉 AWS DLC deployment completed successfully!"
echo ""
echo "Benefits of using AWS Deep Learning Containers:"
echo "✅ Pre-optimized ML frameworks"
echo "✅ AWS-specific performance tunings"
echo "✅ Regular security updates"
echo "✅ Consistent versioning"
echo "✅ Reduced build times"
echo ""
echo "To test the deployment:"
echo "1. Port forward: kubectl port-forward service/$SERVICE_NAME 8000:8000"
echo "2. Run test client: python test_client.py"
echo ""
echo "To check logs: kubectl logs -l app=$DEPLOYMENT_NAME -f"

# Show AWS DLC specific information
echo ""
echo "📊 AWS DLC Information:"
echo "Base Image: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
if [ "$DEPLOYMENT_TYPE" = "vllm" ]; then
    echo "DLC Type: PyTorch Inference GPU"
    echo "Framework: PyTorch 2.1.0 + CUDA 12.1"
elif [ "$DEPLOYMENT_TYPE" = "neuron" ]; then
    echo "DLC Type: PyTorch Neuron Inference"
    echo "Framework: PyTorch 2.1.2 + Neuron SDK 2.18.2"
fi
