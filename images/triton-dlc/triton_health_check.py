#!/usr/bin/env python3
"""
Health check script for Triton-compatible server
"""

import requests
import sys
import os

def check_health():
    """Check if the Triton-compatible server is healthy"""
    base_url = f"http://localhost:{os.getenv('PORT', '8000')}"
    
    try:
        # Check readiness
        response = requests.get(f"{base_url}/v2/health/ready", timeout=10)
        if response.status_code != 200:
            print(f"Readiness check failed: {response.status_code}")
            return False
        
        # Check liveness
        response = requests.get(f"{base_url}/v2/health/live", timeout=10)
        if response.status_code != 200:
            print(f"Liveness check failed: {response.status_code}")
            return False
        
        print("Health check passed")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")
        return False

if __name__ == "__main__":
    if check_health():
        sys.exit(0)
    else:
        sys.exit(1)
