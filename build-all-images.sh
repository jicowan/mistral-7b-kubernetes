#!/bin/bash

# Build and push all container images to ECR
# Usage: ./build-all-images.sh [AWS_REGION] [AWS_ACCOUNT_ID]

set -e

# Configuration
AWS_REGION=${1:-"us-west-2"}
AWS_ACCOUNT_ID=${2:-$(aws sts get-caller-identity --query Account --output text)}
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Image configurations
declare -A IMAGES=(
    ["vllm-mistral-7b"]="Dockerfile"
    ["triton-vllm-mistral-7b"]="Dockerfile.triton"
    ["neuron-mistral-7b"]="Dockerfile.neuron"
    ["vllm-mistral-7b-dlc"]="aws-dlc/Dockerfile.vllm-dlc"
    ["neuron-mistral-7b-dlc"]="aws-dlc/Dockerfile.neuron-dlc"
    ["triton-mistral-7b-dlc"]="aws-dlc/Dockerfile.triton-complete"
)

echo "🚀 Building and Pushing All Mistral 7B Images to ECR"
echo "=================================================="
echo "AWS Region: $AWS_REGION"
echo "AWS Account: $AWS_ACCOUNT_ID"
echo "ECR Registry: $ECR_REGISTRY"
echo ""

# Authenticate with ECR
echo "1. Authenticating with ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY

# Create ECR repositories if they don't exist
echo ""
echo "2. Creating ECR repositories..."
for image_name in "${!IMAGES[@]}"; do
    echo "   Creating repository: $image_name"
    aws ecr create-repository --repository-name $image_name --region $AWS_REGION 2>/dev/null || echo "   Repository $image_name already exists"
done

# Build and push images
echo ""
echo "3. Building and pushing images..."
for image_name in "${!IMAGES[@]}"; do
    dockerfile="${IMAGES[$image_name]}"
    full_image_name="${ECR_REGISTRY}/${image_name}:latest"
    
    echo ""
    echo "   Building $image_name..."
    echo "   Dockerfile: $dockerfile"
    echo "   Target: $full_image_name"
    
    # Determine build context based on dockerfile location
    if [[ $dockerfile == aws-dlc/* ]]; then
        build_context="."
        dockerfile_path="$dockerfile"
    else
        build_context="."
        dockerfile_path="$dockerfile"
    fi
    
    # Build image
    docker build -f $dockerfile_path -t $image_name:latest -t $full_image_name $build_context
    
    if [ $? -eq 0 ]; then
        echo "   ✅ Build successful for $image_name"
        
        # Push to ECR
        echo "   Pushing to ECR..."
        docker push $full_image_name
        
        if [ $? -eq 0 ]; then
            echo "   ✅ Push successful for $image_name"
        else
            echo "   ❌ Push failed for $image_name"
        fi
    else
        echo "   ❌ Build failed for $image_name"
    fi
done

echo ""
echo "4. Image Summary:"
echo "=================="
for image_name in "${!IMAGES[@]}"; do
    full_image_name="${ECR_REGISTRY}/${image_name}:latest"
    echo "   $full_image_name"
done

echo ""
echo "5. Update Kubernetes deployments:"
echo "================================="
echo "Update your deployment YAML files with these image URIs:"
echo ""
for image_name in "${!IMAGES[@]}"; do
    full_image_name="${ECR_REGISTRY}/${image_name}:latest"
    echo "# For $image_name deployment:"
    echo "image: $full_image_name"
    echo ""
done

echo "🎉 All images built and pushed successfully!"
echo ""
echo "Next steps:"
echo "1. Update your Kubernetes deployment YAML files with the ECR image URIs above"
echo "2. Deploy to your Kubernetes cluster"
echo "3. Test with: kubectl port-forward service/<service-name> 8000:8000"
