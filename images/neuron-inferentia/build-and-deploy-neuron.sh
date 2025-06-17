#!/bin/bash

# Build and Deploy Neuron Mistral 7B to Kubernetes
# Optimized for AWS Inferentia 1 and Inferentia 2

set -e

# Configuration
IMAGE_NAME="neuron-mistral-7b"
IMAGE_TAG="latest"
REGISTRY="" # Set your container registry here
INSTANCE_TYPE="inf1.xlarge" # or inf2.xlarge

echo "🚀 Building and Deploying Neuron Mistral 7B Server"
echo "=================================================="

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
echo "2. Building Docker image for Neuron..."
docker build -f Dockerfile.neuron -t ${IMAGE_NAME}:${IMAGE_TAG} .

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
        sed -i.bak "s|image: ${IMAGE_NAME}:${IMAGE_TAG}|image: ${REGISTRY}${IMAGE_NAME}:${IMAGE_TAG}|g" kubernetes-deployment-neuron.yaml
    else
        echo "❌ Failed to push to registry"
        exit 1
    fi
else
    echo ""
    echo "3. Skipping registry push (REGISTRY not set)"
fi

# Check if Neuron device plugin is installed
echo ""
echo "4. Checking Neuron device plugin..."
kubectl get daemonset -n kube-system | grep -q neuron-device-plugin
if [ $? -eq 0 ]; then
    echo "✅ Neuron device plugin detected"
else
    echo "⚠️  Neuron device plugin not detected. Installing..."
    kubectl apply -f https://raw.githubusercontent.com/aws-neuron/aws-neuron-sdk/master/src/k8/k8s-neuron-device-plugin.yml
    
    # Wait for device plugin to be ready
    kubectl rollout status daemonset/neuron-device-plugin -n kube-system --timeout=300s
    
    if [ $? -eq 0 ]; then
        echo "✅ Neuron device plugin installed successfully"
    else
        echo "❌ Failed to install Neuron device plugin"
        exit 1
    fi
fi

# Update deployment with correct instance type
echo ""
echo "5. Configuring deployment for ${INSTANCE_TYPE}..."
sed -i.bak "s|node.kubernetes.io/instance-type: inf1.xlarge|node.kubernetes.io/instance-type: ${INSTANCE_TYPE}|g" kubernetes-deployment-neuron.yaml

# Deploy to Kubernetes
echo ""
echo "6. Deploying to Kubernetes..."
kubectl apply -f kubernetes-deployment-neuron.yaml

if [ $? -eq 0 ]; then
    echo "✅ Deployment applied successfully"
else
    echo "❌ Deployment failed"
    exit 1
fi

# Wait for deployment to be ready (longer timeout due to compilation)
echo ""
echo "7. Waiting for deployment to be ready (this may take 10-20 minutes due to model compilation)..."
kubectl wait --for=condition=available --timeout=1200s deployment/neuron-mistral-7b

if [ $? -eq 0 ]; then
    echo "✅ Deployment is ready"
else
    echo "❌ Deployment failed to become ready within timeout"
    echo "Check logs with: kubectl logs -l app=neuron-mistral-7b"
    exit 1
fi

# Get service information
echo ""
echo "8. Service Information:"
kubectl get service neuron-mistral-7b-service

# Show pod status
echo ""
echo "9. Pod Status:"
kubectl get pods -l app=neuron-mistral-7b

# Check Neuron utilization
echo ""
echo "10. Neuron Device Status:"
kubectl exec -it $(kubectl get pods -l app=neuron-mistral-7b -o jsonpath='{.items[0].metadata.name}') -- neuron-ls || echo "Could not check neuron status"

# Show logs
echo ""
echo "11. Recent Logs:"
kubectl logs -l app=neuron-mistral-7b --tail=20

echo ""
echo "🎉 Neuron deployment completed successfully!"
echo ""
echo "⚠️  Note: First inference may be slow due to model compilation"
echo ""
echo "To test the deployment:"
echo "1. Port forward: kubectl port-forward service/neuron-mistral-7b-service 8000:8000"
echo "2. Run test client: python test_client.py"
echo ""
echo "To check Neuron utilization: kubectl exec -it <pod-name> -- neuron-top"
echo "To check logs: kubectl logs -l app=neuron-mistral-7b -f"
echo ""
echo "Instance types supported:"
echo "- Inferentia 1: inf1.xlarge, inf1.2xlarge, inf1.6xlarge, inf1.24xlarge"
echo "- Inferentia 2: inf2.xlarge, inf2.8xlarge, inf2.24xlarge, inf2.48xlarge"
