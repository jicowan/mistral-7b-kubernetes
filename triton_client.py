#!/usr/bin/env python3
"""
Triton client for vLLM Mistral 7B server
"""

import numpy as np
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException
import time

class TritonVLLMClient:
    def __init__(self, url: str = "localhost:8000", protocol: str = "http"):
        """Initialize Triton client
        
        Args:
            url: Triton server URL
            protocol: 'http' or 'grpc'
        """
        self.url = url
        self.protocol = protocol
        
        if protocol == "http":
            self.client = httpclient.InferenceServerClient(url=f"http://{url}")
        else:
            self.client = grpcclient.InferenceServerClient(url=url)
    
    def is_server_ready(self):
        """Check if server is ready"""
        try:
            return self.client.is_server_ready()
        except InferenceServerException as e:
            print(f"Server not ready: {e}")
            return False
    
    def is_model_ready(self, model_name: str = "vllm_mistral"):
        """Check if model is ready"""
        try:
            return self.client.is_model_ready(model_name)
        except InferenceServerException as e:
            print(f"Model not ready: {e}")
            return False
    
    def get_model_metadata(self, model_name: str = "vllm_mistral"):
        """Get model metadata"""
        try:
            return self.client.get_model_metadata(model_name)
        except InferenceServerException as e:
            print(f"Failed to get metadata: {e}")
            return None
    
    def generate(self, prompt: str, max_tokens: int = 512, 
                temperature: float = 0.7, top_p: float = 0.9,
                model_name: str = "vllm_mistral"):
        """Generate text using Triton server"""
        
        # Prepare inputs
        inputs = []
        
        # Prompt input
        prompt_data = np.array([prompt.encode('utf-8')], dtype=object)
        inputs.append(httpclient.InferInput("prompt", prompt_data.shape, "BYTES"))
        inputs[-1].set_data_from_numpy(prompt_data)
        
        # Max tokens input
        max_tokens_data = np.array([max_tokens], dtype=np.int32)
        inputs.append(httpclient.InferInput("max_tokens", max_tokens_data.shape, "INT32"))
        inputs[-1].set_data_from_numpy(max_tokens_data)
        
        # Temperature input
        temperature_data = np.array([temperature], dtype=np.float32)
        inputs.append(httpclient.InferInput("temperature", temperature_data.shape, "FP32"))
        inputs[-1].set_data_from_numpy(temperature_data)
        
        # Top-p input
        top_p_data = np.array([top_p], dtype=np.float32)
        inputs.append(httpclient.InferInput("top_p", top_p_data.shape, "FP32"))
        inputs[-1].set_data_from_numpy(top_p_data)
        
        # Prepare outputs
        outputs = [httpclient.InferRequestedOutput("generated_text")]
        
        # Make inference request
        try:
            response = self.client.infer(model_name, inputs, outputs=outputs)
            result = response.as_numpy("generated_text")
            return result[0].decode('utf-8')
        except InferenceServerException as e:
            print(f"Inference failed: {e}")
            return None

def main():
    # Initialize client
    client = TritonVLLMClient()
    
    print("🚀 Testing Triton vLLM Mistral 7B Server")
    print("=" * 50)
    
    # Check server status
    print("1. Server Status...")
    if not client.is_server_ready():
        print("❌ Server not ready")
        return
    print("✅ Server is ready")
    
    # Check model status
    print("\n2. Model Status...")
    if not client.is_model_ready():
        print("❌ Model not ready")
        return
    print("✅ Model is ready")
    
    # Get model metadata
    print("\n3. Model Metadata...")
    metadata = client.get_model_metadata()
    if metadata:
        print(f"   Model: {metadata.name}")
        print(f"   Platform: {metadata.platform}")
        print(f"   Versions: {metadata.versions}")
    
    # Test generation
    print("\n4. Text Generation Tests...")
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to calculate fibonacci numbers.",
        "What are the benefits of using Kubernetes?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n   Test {i}: {prompt}")
        print("   " + "-" * 40)
        
        start_time = time.time()
        result = client.generate(
            prompt=prompt,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9
        )
        end_time = time.time()
        
        if result:
            print(f"   Response: {result[:200]}...")
            print(f"   Time: {end_time - start_time:.2f}s")
        else:
            print("   ❌ Generation failed")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()
