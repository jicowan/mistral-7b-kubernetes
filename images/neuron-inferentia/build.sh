#!/bin/bash

# Build Neuron Inferentia Image
set -e

IMAGE_NAME="neuron-mistral-7b"
IMAGE_TAG=${1:-"latest"}
REGISTRY=${2:-""}

echo "🚀 Building Neuron Inferentia Image"
echo "==================================="
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
echo "🎉 Neuron Inferentia image ready!"
echo "To test: docker run -p 8000:8000 $IMAGE_NAME:$IMAGE_TAG"
