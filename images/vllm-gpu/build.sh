#!/bin/bash

# Build vLLM GPU Image
set -e

IMAGE_NAME="vllm-mistral-7b"
IMAGE_TAG=${1:-"latest"}
REGISTRY=${2:-""}

echo "🚀 Building vLLM GPU Image"
echo "=========================="
echo "Image: $IMAGE_NAME:$IMAGE_TAG"
echo "Registry: ${REGISTRY:-"(local only)"}"

# Build the image
docker build -t $IMAGE_NAME:$IMAGE_TAG .

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    
    # Tag and push if registry provided
    if [ ! -z "$REGISTRY" ]; then
        FULL_IMAGE_NAME="$REGISTRY/$IMAGE_NAME:$IMAGE_TAG"
        docker tag $IMAGE_NAME:$IMAGE_TAG $FULL_IMAGE_NAME
        docker push $FULL_IMAGE_NAME
        echo "✅ Pushed to $FULL_IMAGE_NAME"
    fi
else
    echo "❌ Build failed!"
    exit 1
fi

echo ""
echo "🎉 vLLM GPU image ready!"
echo "To test: docker run -p 8000:8000 --gpus all $IMAGE_NAME:$IMAGE_TAG"
