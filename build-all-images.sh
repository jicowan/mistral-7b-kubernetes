#!/bin/bash

# Build and push container images to ECR - Reorganized Version
# Usage: ./build-all-images.sh [IMAGE_NAME] [AWS_REGION] [AWS_ACCOUNT_ID]
# 
# IMAGE_NAME options:
#   all (default)     - Build all images
#   vllm-gpu         - vLLM with NVIDIA GPUs
#   triton-gpu       - Triton with NVIDIA GPUs  
#   neuron-inferentia - Neuron with AWS Inferentia
#   vllm-dlc         - vLLM with AWS DLC
#   triton-dlc       - Triton with AWS DLC
#   neuron-dlc       - Neuron with AWS DLC

set -e

# Configuration
IMAGE_TO_BUILD=${1:-"all"}
AWS_REGION=${2:-"us-west-2"}
AWS_ACCOUNT_ID=${3:-$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "")}
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Image configurations
declare -A IMAGES=(
    ["vllm-gpu"]="vllm-mistral-7b"
    ["triton-gpu"]="triton-vllm-mistral-7b"
    ["neuron-inferentia"]="neuron-mistral-7b"
    ["vllm-dlc"]="vllm-mistral-7b-dlc"
    ["triton-dlc"]="triton-mistral-7b-dlc"
    ["neuron-dlc"]="neuron-mistral-7b-dlc"
)

echo "🚀 Building Mistral 7B Container Images"
echo "======================================="
echo "Target: $IMAGE_TO_BUILD"
echo "AWS Region: $AWS_REGION"
echo "AWS Account: ${AWS_ACCOUNT_ID:-"(not detected)"}"
echo "ECR Registry: ${ECR_REGISTRY:-"(local build only)"}"
echo ""

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

if [ "$AWS_ACCOUNT_ID" != "" ] && ! command_exists aws; then
    echo "❌ AWS CLI is not installed but AWS Account ID provided"
    exit 1
fi

echo "✅ Prerequisites check passed"

# Authenticate with ECR if AWS credentials available
if [ "$AWS_ACCOUNT_ID" != "" ]; then
    echo ""
    echo "2. Authenticating with ECR..."
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REGISTRY
    
    if [ $? -eq 0 ]; then
        echo "✅ ECR authentication successful"
    else
        echo "❌ ECR authentication failed"
        exit 1
    fi
    
    # Create ECR repositories
    echo ""
    echo "3. Creating ECR repositories..."
    if [ "$IMAGE_TO_BUILD" = "all" ]; then
        for image_key in "${!IMAGES[@]}"; do
            image_name="${IMAGES[$image_key]}"
            echo "   Creating repository: $image_name"
            aws ecr create-repository --repository-name $image_name --region $AWS_REGION 2>/dev/null || echo "   Repository $image_name already exists"
        done
    else
        image_name="${IMAGES[$IMAGE_TO_BUILD]}"
        if [ "$image_name" != "" ]; then
            echo "   Creating repository: $image_name"
            aws ecr create-repository --repository-name $image_name --region $AWS_REGION 2>/dev/null || echo "   Repository $image_name already exists"
        fi
    fi
else
    echo ""
    echo "2. Skipping ECR authentication (building locally only)"
fi

# Build function
build_image() {
    local image_key=$1
    local image_name="${IMAGES[$image_key]}"
    local image_dir="images/$image_key"
    
    if [ ! -d "$image_dir" ]; then
        echo "❌ Directory $image_dir not found"
        return 1
    fi
    
    echo ""
    echo "Building $image_key ($image_name)..."
    echo "Directory: $image_dir"
    echo "================================"
    
    cd "$image_dir"
    
    # Determine registry parameter
    local registry_param=""
    if [ "$AWS_ACCOUNT_ID" != "" ]; then
        registry_param="$ECR_REGISTRY"
    fi
    
    # Run the individual build script
    ./build.sh latest "$registry_param"
    
    if [ $? -eq 0 ]; then
        echo "✅ $image_key build completed successfully"
    else
        echo "❌ $image_key build failed"
        cd - > /dev/null
        return 1
    fi
    
    cd - > /dev/null
}

# Build images
echo ""
echo "4. Building images..."

if [ "$IMAGE_TO_BUILD" = "all" ]; then
    echo "Building all images..."
    for image_key in "${!IMAGES[@]}"; do
        build_image "$image_key"
    done
else
    if [ "${IMAGES[$IMAGE_TO_BUILD]}" = "" ]; then
        echo "❌ Unknown image: $IMAGE_TO_BUILD"
        echo "Available options: ${!IMAGES[@]} all"
        exit 1
    fi
    
    echo "Building single image: $IMAGE_TO_BUILD"
    build_image "$IMAGE_TO_BUILD"
fi

echo ""
echo "5. Build Summary:"
echo "================="
if [ "$IMAGE_TO_BUILD" = "all" ]; then
    for image_key in "${!IMAGES[@]}"; do
        image_name="${IMAGES[$image_key]}"
        if [ "$AWS_ACCOUNT_ID" != "" ]; then
            echo "   $ECR_REGISTRY/$image_name:latest"
        else
            echo "   $image_name:latest (local)"
        fi
    done
else
    image_name="${IMAGES[$IMAGE_TO_BUILD]}"
    if [ "$AWS_ACCOUNT_ID" != "" ]; then
        echo "   $ECR_REGISTRY/$image_name:latest"
    else
        echo "   $image_name:latest (local)"
    fi
fi

echo ""
echo "🎉 Build process completed!"
echo ""
echo "Usage examples:"
echo "==============="
echo "# Build all images:"
echo "./build-all-images.sh all"
echo ""
echo "# Build specific image:"
echo "./build-all-images.sh vllm-gpu"
echo "./build-all-images.sh triton-dlc"
echo ""
echo "# Build and push to ECR:"
echo "./build-all-images.sh vllm-gpu us-west-2 123456789012"
echo ""
echo "# Individual builds:"
echo "cd images/vllm-gpu && ./build.sh"
echo "cd images/triton-dlc && ./build.sh latest your-registry"
