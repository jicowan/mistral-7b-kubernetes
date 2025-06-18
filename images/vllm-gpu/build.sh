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
fi

echo ""
echo "🎉 vLLM GPU image ready!"
echo "To test: docker run -p 8000:8000 --gpus all $IMAGE_NAME:$IMAGE_TAG"
