#!/bin/bash

# Update Kubernetes deployment files with ECR image URIs - Reorganized Version
# Usage: ./update-k8s-images.sh [IMAGE_NAME] [AWS_REGION] [AWS_ACCOUNT_ID]
# 
# IMAGE_NAME options:
#   all (default)     - Update all deployment files
#   vllm-gpu         - Update vLLM GPU deployment
#   triton-gpu       - Update Triton GPU deployment
#   neuron-inferentia - Update Neuron Inferentia deployment
#   vllm-dlc         - Update vLLM DLC deployment
#   triton-dlc       - Update Triton DLC deployment
#   neuron-dlc       - Update Neuron DLC deployment

set -e

# Configuration
IMAGE_TO_UPDATE=${1:-"all"}
AWS_REGION=${2:-"us-west-2"}
AWS_ACCOUNT_ID=${3:-$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "")}
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

echo "🔄 Updating Kubernetes deployment files with ECR image URIs"
echo "=========================================================="
echo "Target: $IMAGE_TO_UPDATE"
echo "AWS Region: $AWS_REGION"
echo "AWS Account: ${AWS_ACCOUNT_ID:-"(not detected)"}"
echo "ECR Registry: ${ECR_REGISTRY:-"(local images only)"}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to get ECR image name for directory
get_image_name() {
    case "$1" in
        "vllm-gpu") echo "vllm-mistral-7b" ;;
        "triton-gpu") echo "triton-vllm-mistral-7b" ;;
        "neuron-inferentia") echo "neuron-mistral-7b" ;;
        "vllm-dlc") echo "vllm-mistral-7b-dlc" ;;
        "triton-dlc") echo "triton-mistral-7b-dlc" ;;
        "neuron-dlc") echo "neuron-mistral-7b-dlc" ;;
        *) echo "" ;;
    esac
}

# Function to get all image directories
get_all_images() {
    echo "vllm-gpu triton-gpu neuron-inferentia vllm-dlc triton-dlc neuron-dlc"
}

# Check prerequisites
echo "1. Checking prerequisites..."
if [ "$AWS_ACCOUNT_ID" != "" ] && ! command_exists aws; then
    echo "❌ AWS CLI is not installed but AWS Account ID provided"
    exit 1
fi

if [ "$AWS_ACCOUNT_ID" = "" ]; then
    echo "⚠️  No AWS Account ID detected - will use local image names"
fi

echo "✅ Prerequisites check passed"

# Function to update a single deployment file
update_deployment() {
    local image_dir=$1
    local image_name=$(get_image_name "$image_dir")
    local deployment_file="images/$image_dir/kubernetes-deployment.yaml"
    
    if [ -z "$image_name" ]; then
        echo "   ❌ Unknown image directory: $image_dir"
        return 1
    fi
    
    if [ ! -f "$deployment_file" ]; then
        echo "   ⚠️  $deployment_file not found, skipping"
        return 0
    fi
    
    echo "   Updating $image_dir deployment..."
    echo "     File: $deployment_file"
    echo "     Image: $image_name"
    
    # Determine the image URI to use
    local image_uri
    if [ "$AWS_ACCOUNT_ID" != "" ]; then
        image_uri="${ECR_REGISTRY}/${image_name}:latest"
    else
        image_uri="${image_name}:latest"
    fi
    
    # Create backup
    cp "$deployment_file" "${deployment_file}.bak"
    
    # Update the image reference
    # Handle different possible image reference patterns
    sed -i.tmp \
        -e "s|image: ${image_name}:latest|image: ${image_uri}|g" \
        -e "s|image: ${image_name}$|image: ${image_uri}|g" \
        -e "s|image: .*/${image_name}:latest|image: ${image_uri}|g" \
        -e "s|image: .*/${image_name}$|image: ${image_uri}|g" \
        "$deployment_file"
    
    # Remove temporary file
    rm "${deployment_file}.tmp"
    
    echo "     ✅ Updated to: $image_uri"
    return 0
}

# Update deployments
echo ""
echo "2. Updating deployment files..."

if [ "$IMAGE_TO_UPDATE" = "all" ]; then
    echo "Updating all deployment files..."
    for image_dir in $(get_all_images); do
        update_deployment "$image_dir"
    done
else
    image_name=$(get_image_name "$IMAGE_TO_UPDATE")
    if [ -z "$image_name" ]; then
        echo "❌ Unknown image: $IMAGE_TO_UPDATE"
        echo "Available options: $(get_all_images) all"
        exit 1
    fi
    
    echo "Updating single deployment: $IMAGE_TO_UPDATE"
    update_deployment "$IMAGE_TO_UPDATE"
fi

echo ""
echo "3. Updated Image URIs:"
echo "====================="

if [ "$IMAGE_TO_UPDATE" = "all" ]; then
    for image_dir in $(get_all_images); do
        image_name=$(get_image_name "$image_dir")
        if [ "$AWS_ACCOUNT_ID" != "" ]; then
            echo "   $image_dir: ${ECR_REGISTRY}/${image_name}:latest"
        else
            echo "   $image_dir: ${image_name}:latest (local)"
        fi
    done
else
    image_name=$(get_image_name "$IMAGE_TO_UPDATE")
    if [ "$AWS_ACCOUNT_ID" != "" ]; then
        echo "   $IMAGE_TO_UPDATE: ${ECR_REGISTRY}/${image_name}:latest"
    else
        echo "   $IMAGE_TO_UPDATE: ${image_name}:latest (local)"
    fi
fi

echo ""
echo "4. Deployment Commands:"
echo "======================"

if [ "$IMAGE_TO_UPDATE" = "all" ]; then
    echo "Deploy all images:"
    for image_dir in $(get_all_images); do
        echo "   kubectl apply -f images/$image_dir/kubernetes-deployment.yaml"
    done
else
    echo "Deploy $IMAGE_TO_UPDATE:"
    echo "   kubectl apply -f images/$IMAGE_TO_UPDATE/kubernetes-deployment.yaml"
fi

echo ""
echo "✅ Kubernetes deployment files updated!"
echo ""
echo "📁 Backup files created with .bak extension"
echo ""
echo "Usage examples:"
echo "==============="
echo "# Update all deployments:"
echo "./update-k8s-images.sh all"
echo ""
echo "# Update specific deployment:"
echo "./update-k8s-images.sh vllm-gpu"
echo "./update-k8s-images.sh triton-dlc"
echo ""
echo "# Update with specific AWS account:"
echo "./update-k8s-images.sh vllm-gpu us-west-2 123456789012"
echo ""
echo "# Deploy updated manifests:"
echo "kubectl apply -f images/vllm-gpu/kubernetes-deployment.yaml"
echo "kubectl apply -f images/triton-dlc/kubernetes-deployment.yaml"
