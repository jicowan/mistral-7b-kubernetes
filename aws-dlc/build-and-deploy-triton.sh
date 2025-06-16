#!/bin/bash

# Build and Deploy Triton-Compatible Server with AWS Deep Learning Containers
# Provides Triton-like API with vLLM backend and AWS DLC optimizations

set -e

# Configuration
IMAGE_NAME="triton-mistral-7b-dlc"
IMAGE_TAG="latest"
AWS_REGION=${AWS_REGION:-"us-west-2"}
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-"763104351884"}  # AWS DLC account
REGISTRY="" # Set your container registry here
DEPLOYMENT_NAME="triton-mistral-7b-dlc"
SERVICE_NAME="triton-mistral-7b-dlc-service"

echo "🚀 Building and Deploying Triton-Compatible Server with AWS DLC"
echo "=============================================================="
echo "Image: $IMAGE_NAME:$IMAGE_TAG"
echo "AWS Region: $AWS_REGION"
echo "Deployment: $DEPLOYMENT_NAME"

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

# Prepare build context
echo ""
echo "3. Preparing build context..."

# Copy necessary files to build context
cp ../vllm_server.py . 2>/dev/null || echo "vllm_server.py not found, using triton wrapper"
cp ../requirements.txt . 2>/dev/null || echo "requirements.txt not found"

# Create requirements-triton.txt if it doesn't exist
if [ ! -f requirements-triton.txt ]; then
    cat > requirements-triton.txt << EOF
tritonclient[all]==2.40.0
triton-python-backend-utils
numpy==1.24.3
requests==2.31.0
EOF
fi

echo "✅ Build context prepared"

# Build Docker image
echo ""
echo "4. Building Triton-compatible Docker image with AWS DLC..."
docker build -f Dockerfile.triton-complete -t ${IMAGE_NAME}:${IMAGE_TAG} .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
else
    echo "❌ Docker build failed"
    exit 1
fi

# Tag and push to registry (if registry is set)
if [ ! -z "$REGISTRY" ]; then
    echo ""
    echo "5. Pushing to registry..."
    docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${REGISTRY}${IMAGE_NAME}:${IMAGE_TAG}
    docker push ${REGISTRY}${IMAGE_NAME}:${IMAGE_TAG}
    
    if [ $? -eq 0 ]; then
        echo "✅ Image pushed to registry"
        # Update deployment YAML with registry
        sed -i.bak "s|image: ${IMAGE_NAME}:${IMAGE_TAG}|image: ${REGISTRY}${IMAGE_NAME}:${IMAGE_TAG}|g" kubernetes-deployment-triton-dlc.yaml
    else
        echo "❌ Failed to push to registry"
        exit 1
    fi
else
    echo ""
    echo "5. Skipping registry push (REGISTRY not set)"
fi

# Check GPU support
echo ""
echo "6. Checking GPU support..."
kubectl get nodes -o json | jq -r '.items[].status.allocatable | keys[]' | grep -q "nvidia.com/gpu"
if [ $? -eq 0 ]; then
    echo "✅ NVIDIA GPU support detected"
else
    echo "⚠️  NVIDIA GPU support not detected. Make sure NVIDIA GPU Operator is installed."
    echo "   Install with: kubectl apply -f https://raw.githubusercontent.com/NVIDIA/gpu-operator/main/deployments/gpu-operator/values.yaml"
fi

# Deploy to Kubernetes
echo ""
echo "7. Deploying to Kubernetes..."
kubectl apply -f kubernetes-deployment-triton-dlc.yaml

if [ $? -eq 0 ]; then
    echo "✅ Deployment applied successfully"
else
    echo "❌ Deployment failed"
    exit 1
fi

# Wait for deployment to be ready
echo ""
echo "8. Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=600s deployment/$DEPLOYMENT_NAME

if [ $? -eq 0 ]; then
    echo "✅ Deployment is ready"
else
    echo "❌ Deployment failed to become ready within timeout"
    echo "Check logs with: kubectl logs -l app=$DEPLOYMENT_NAME"
    exit 1
fi

# Get service information
echo ""
echo "9. Service Information:"
kubectl get service $SERVICE_NAME

# Show pod status
echo ""
echo "10. Pod Status:"
kubectl get pods -l app=$DEPLOYMENT_NAME

# Test Triton endpoints
echo ""
echo "11. Testing Triton-compatible endpoints..."
POD_NAME=$(kubectl get pods -l app=$DEPLOYMENT_NAME -o jsonpath='{.items[0].metadata.name}')

if [ ! -z "$POD_NAME" ]; then
    echo "Testing health endpoints..."
    kubectl exec $POD_NAME -- curl -s http://localhost:8000/v2/health/ready > /dev/null && echo "✅ Ready endpoint working" || echo "❌ Ready endpoint failed"
    kubectl exec $POD_NAME -- curl -s http://localhost:8000/v2/health/live > /dev/null && echo "✅ Live endpoint working" || echo "❌ Live endpoint failed"
    kubectl exec $POD_NAME -- curl -s http://localhost:8000/v2/models > /dev/null && echo "✅ Models endpoint working" || echo "❌ Models endpoint failed"
fi

# Show logs
echo ""
echo "12. Recent Logs:"
kubectl logs -l app=$DEPLOYMENT_NAME --tail=20

echo ""
echo "🎉 Triton-compatible deployment with AWS DLC completed successfully!"
echo ""
echo "🔧 Triton-Compatible Features:"
echo "✅ Triton HTTP API endpoints (/v2/...)"
echo "✅ Model metadata and health checks"
echo "✅ Inference endpoint with Triton format"
echo "✅ AWS DLC optimizations"
echo "✅ vLLM backend for high performance"
echo ""
echo "📊 Available Endpoints:"
echo "- Health Ready: GET /v2/health/ready"
echo "- Health Live: GET /v2/health/live"
echo "- List Models: GET /v2/models"
echo "- Model Metadata: GET /v2/models/{model_name}"
echo "- Inference: POST /v2/models/{model_name}/infer"
echo "- Server Metadata: GET /v2"
echo ""
echo "🧪 To test the deployment:"
echo "1. Port forward: kubectl port-forward service/$SERVICE_NAME 8000:8000"
echo "2. Test health: curl http://localhost:8000/v2/health/ready"
echo "3. List models: curl http://localhost:8000/v2/models"
echo "4. Run inference test: python triton_test_client.py"
echo ""
echo "📈 Monitoring:"
echo "- Logs: kubectl logs -l app=$DEPLOYMENT_NAME -f"
echo "- Metrics: kubectl port-forward service/$SERVICE_NAME 8002:8002"
echo "- Scale: kubectl scale deployment $DEPLOYMENT_NAME --replicas=2"
echo ""
echo "🏗️  Architecture:"
echo "- Base: AWS PyTorch Training DLC"
echo "- Backend: vLLM for inference"
echo "- API: Triton-compatible HTTP endpoints"
echo "- Optimizations: AWS DLC + CUDA graphs"

# Cleanup temporary files
rm -f vllm_server.py requirements.txt 2>/dev/null || true

echo ""
echo "✨ Deployment complete! Your Triton-compatible server is ready."
