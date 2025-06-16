#!/bin/bash

# Update Kubernetes deployment files with ECR image URIs
# Usage: ./update-k8s-images.sh [AWS_REGION] [AWS_ACCOUNT_ID]

set -e

# Configuration
AWS_REGION=${1:-"us-west-2"}
AWS_ACCOUNT_ID=${2:-$(aws sts get-caller-identity --query Account --output text)}
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "🔄 Updating Kubernetes deployment files with ECR image URIs"
echo "=========================================================="
echo "ECR Registry: $ECR_REGISTRY"
echo ""

# Update standard deployments
echo "1. Updating standard deployment files..."

# vLLM deployment
sed -i.bak "s|image: vllm-mistral-7b:latest|image: ${ECR_REGISTRY}/vllm-mistral-7b:latest|g" kubernetes-deployment.yaml
echo "   ✅ Updated kubernetes-deployment.yaml"

# Triton deployment
sed -i.bak "s|image: triton-vllm-mistral-7b:latest|image: ${ECR_REGISTRY}/triton-vllm-mistral-7b:latest|g" kubernetes-deployment-triton.yaml
echo "   ✅ Updated kubernetes-deployment-triton.yaml"

# Neuron deployment
sed -i.bak "s|image: neuron-mistral-7b:latest|image: ${ECR_REGISTRY}/neuron-mistral-7b:latest|g" kubernetes-deployment-neuron.yaml
echo "   ✅ Updated kubernetes-deployment-neuron.yaml"

# Update AWS DLC deployments
echo ""
echo "2. Updating AWS DLC deployment files..."

# vLLM DLC deployment
sed -i.bak "s|image: vllm-mistral-7b-dlc:latest|image: ${ECR_REGISTRY}/vllm-mistral-7b-dlc:latest|g" aws-dlc/kubernetes-deployment-vllm-dlc.yaml
echo "   ✅ Updated aws-dlc/kubernetes-deployment-vllm-dlc.yaml"

# Neuron DLC deployment
sed -i.bak "s|image: neuron-mistral-7b-dlc:latest|image: ${ECR_REGISTRY}/neuron-mistral-7b-dlc:latest|g" aws-dlc/kubernetes-deployment-neuron-dlc.yaml
echo "   ✅ Updated aws-dlc/kubernetes-deployment-neuron-dlc.yaml"

# Triton DLC deployment
sed -i.bak "s|image: triton-mistral-7b-dlc:latest|image: ${ECR_REGISTRY}/triton-mistral-7b-dlc:latest|g" aws-dlc/kubernetes-deployment-triton-dlc.yaml
echo "   ✅ Updated aws-dlc/kubernetes-deployment-triton-dlc.yaml"

echo ""
echo "3. Updated Image URIs:"
echo "====================="
echo "Standard Images:"
echo "   ${ECR_REGISTRY}/vllm-mistral-7b:latest"
echo "   ${ECR_REGISTRY}/triton-vllm-mistral-7b:latest"
echo "   ${ECR_REGISTRY}/neuron-mistral-7b:latest"
echo ""
echo "AWS DLC Images:"
echo "   ${ECR_REGISTRY}/vllm-mistral-7b-dlc:latest"
echo "   ${ECR_REGISTRY}/neuron-mistral-7b-dlc:latest"
echo "   ${ECR_REGISTRY}/triton-mistral-7b-dlc:latest"

echo ""
echo "✅ All Kubernetes deployment files updated!"
echo ""
echo "Backup files created with .bak extension"
echo "You can now deploy with: kubectl apply -f <deployment-file>.yaml"
