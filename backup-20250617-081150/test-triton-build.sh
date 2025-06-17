#!/bin/bash

# Test script to verify Triton build works
set -e

echo "🧪 Testing Triton Docker build..."
echo "================================"

# Test the build context and file paths
echo "1. Checking required files exist..."

# Check files from root directory perspective
files_to_check=(
    "aws-dlc/Dockerfile.triton-complete"
    "aws-dlc/triton_server_wrapper.py"
    "aws-dlc/triton_health_check.py"
    "aws-dlc/requirements-triton.txt"
    "triton-model-repository/vllm_mistral/config.pbtxt"
    "triton-model-repository/vllm_mistral/1/model.py"
)

for file in "${files_to_check[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file (MISSING)"
        exit 1
    fi
done

echo ""
echo "2. Testing Docker build..."
docker build -f aws-dlc/Dockerfile.triton-complete -t triton-test:latest .

if [ $? -eq 0 ]; then
    echo "   ✅ Docker build successful!"
    
    echo ""
    echo "3. Checking image contents..."
    docker run --rm triton-test:latest ls -la /app/
    
    echo ""
    echo "4. Checking model repository..."
    docker run --rm triton-test:latest ls -la /models/
    
    echo ""
    echo "✅ All tests passed! Triton build is working correctly."
else
    echo "   ❌ Docker build failed!"
    exit 1
fi
