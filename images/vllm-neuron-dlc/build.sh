#!/bin/bash

# Build script for vLLM + Neuron DLC Mistral 7B
set -e

# Configuration
IMAGE_NAME="vllm-mistral-7b-neuron"
ECR_REGISTRY="820537372947.dkr.ecr.us-west-2.amazonaws.com"
REGION="us-west-2"

echo "🚀 Building vLLM + Neuron DLC image for Mistral 7B on Inferentia2..."

# Build the image
docker build -t ${IMAGE_NAME}:latest .

# Tag for ECR
docker tag ${IMAGE_NAME}:latest ${ECR_REGISTRY}/${IMAGE_NAME}:latest

echo "✅ Build completed!"
echo "📦 Image: ${ECR_REGISTRY}/${IMAGE_NAME}:latest"
echo ""
echo "To push to ECR:"
echo "aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ECR_REGISTRY}"
echo "docker push ${ECR_REGISTRY}/${IMAGE_NAME}:latest"
