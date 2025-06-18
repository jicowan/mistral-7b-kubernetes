#!/bin/bash

# Build vLLM AWS DLC Image
set -e

IMAGE_NAME="vllm-mistral-7b-dlc"
IMAGE_TAG=${1:-"latest"}
REGISTRY=${2:-""}
AWS_REGION=${AWS_REGION:-"us-west-2"}

echo "🚀 Building vLLM AWS DLC Image"
echo "=============================="
echo "Image: $IMAGE_NAME:$IMAGE_TAG"
echo "Registry: ${REGISTRY:-"(local only)"}"
echo "AWS Region: $AWS_REGION"

# Authenticate with ECR for AWS DLC base image
if command -v aws &> /dev/null; then
    echo "Authenticating with ECR for AWS DLC..."
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin 763104351884.dkr.ecr.$AWS_REGION.amazonaws.com
fi

# Build the image
docker build -t $IMAGE_NAME:$IMAGE_TAG .

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    
    # Tag and push if registry provided
    if [ ! -z "$REGISTRY" ]; then
        FULL_IMAGE_NAME="$REGISTRY/$IMAGE_NAME:$IMAGE_TAG"
        
        echo "🏷️  Tagging image: $IMAGE_NAME:$IMAGE_TAG -> $FULL_IMAGE_NAME"
        docker tag $IMAGE_NAME:$IMAGE_TAG $FULL_IMAGE_NAME
        
        echo "📤 Pushing image: $FULL_IMAGE_NAME"
        docker push $FULL_IMAGE_NAME
        
        if [ $? -eq 0 ]; then
            echo "✅ Successfully pushed to $FULL_IMAGE_NAME"
            
            # Verify the push by checking the image digest
            echo "🔍 Verifying push..."
            docker inspect $FULL_IMAGE_NAME --format='{{.Id}}' 2>/dev/null || echo "⚠️  Could not verify local image ID"
        else
            echo "❌ Failed to push to $FULL_IMAGE_NAME"
            exit 1
        fi
    fi
else
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "🎉 vLLM AWS DLC image ready!"
echo "To test: docker run -p 8000:8000 --gpus all $IMAGE_NAME:$IMAGE_TAG"
