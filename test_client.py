#!/usr/bin/env python3
"""
Test client for vLLM Mistral 7B server
"""

import requests
import json
import time
from typing import Dict, Any

class VLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict[str, Any]:
        """Check server health"""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text from prompt"""
        payload = {
            "prompt": prompt,
            **kwargs
        }
        
        response = requests.post(
            f"{self.base_url}/generate",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> Dict[str, Any]:
        """List available models"""
        response = requests.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()

def main():
    # Initialize client
    client = VLLMClient()
    
    print("🚀 Testing vLLM Mistral 7B Server")
    print("=" * 50)
    
    try:
        # Health check
        print("1. Health Check...")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Model: {health['model']}")
        print(f"   GPU Memory Utilization: {health['gpu_memory_utilization']}")
        print()
        
        # List models
        print("2. Available Models...")
        models = client.list_models()
        for model in models['data']:
            print(f"   - {model['id']}")
        print()
        
        # Test generation
        print("3. Text Generation Test...")
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
            
            print(f"   Response: {result['text'][:200]}...")
            print(f"   Tokens: {result['usage']['total_tokens']}")
            print(f"   Time: {end_time - start_time:.2f}s")
            
            # Calculate tokens per second
            tokens_per_second = result['usage']['completion_tokens'] / (end_time - start_time)
            print(f"   Speed: {tokens_per_second:.1f} tokens/sec")
        
        print("\n✅ All tests completed successfully!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Make sure the server is running.")
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
