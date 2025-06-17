#!/usr/bin/env python3
"""
Triton-compatible server wrapper using AWS DLC + vLLM
Provides Triton-like API endpoints with vLLM backend
"""

import os
import logging
import asyncio
import json
import numpy as np
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from vllm import LLM, SamplingParams

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "32768"))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))

# Global vLLM engine
llm_engine: Optional[LLM] = None

# Triton-compatible request/response models
class TritonInferRequest(BaseModel):
    inputs: List[Dict[str, Any]]
    outputs: Optional[List[Dict[str, str]]] = None
    parameters: Optional[Dict[str, Any]] = None

class TritonInferResponse(BaseModel):
    model_name: str
    model_version: str
    outputs: List[Dict[str, Any]]

class TritonModelMetadata(BaseModel):
    name: str
    versions: List[str]
    platform: str
    inputs: List[Dict[str, Any]]
    outputs: List[Dict[str, Any]]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup the vLLM engine"""
    global llm_engine
    
    logger.info("Starting vLLM engine initialization for Triton wrapper...")
    
    try:
        # Initialize vLLM engine with AWS DLC optimizations
        llm_engine = LLM(
            model=MODEL_NAME,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            trust_remote_code=True,
            dtype="auto",
            enforce_eager=False,  # Use CUDA graphs
            disable_log_stats=False,
        )
        
        logger.info(f"vLLM engine initialized successfully with model: {MODEL_NAME}")
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize vLLM engine: {e}")
        raise
    finally:
        if llm_engine:
            logger.info("Shutting down vLLM engine...")
            del llm_engine

# Create FastAPI app with Triton-compatible endpoints
app = FastAPI(
    title="Triton-Compatible vLLM Server",
    description="Triton-like inference server using vLLM with AWS DLC optimizations",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/v2/health/ready")
async def health_ready():
    """Triton-compatible readiness check"""
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"ready": True}

@app.get("/v2/health/live")
async def health_live():
    """Triton-compatible liveness check"""
    return {"live": True}

@app.get("/v2/models/{model_name}")
async def get_model_metadata(model_name: str):
    """Get model metadata in Triton format"""
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    return TritonModelMetadata(
        name=model_name,
        versions=["1"],
        platform="vllm_python",
        inputs=[
            {
                "name": "prompt",
                "datatype": "BYTES",
                "shape": [1]
            },
            {
                "name": "max_tokens",
                "datatype": "INT32", 
                "shape": [1]
            },
            {
                "name": "temperature",
                "datatype": "FP32",
                "shape": [1]
            }
        ],
        outputs=[
            {
                "name": "generated_text",
                "datatype": "BYTES",
                "shape": [1]
            }
        ]
    )

@app.post("/v2/models/{model_name}/infer")
async def model_infer(model_name: str, request: TritonInferRequest):
    """Triton-compatible inference endpoint"""
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    try:
        # Parse Triton inputs
        inputs_dict = {}
        for input_data in request.inputs:
            name = input_data["name"]
            data = input_data["data"]
            inputs_dict[name] = data[0] if isinstance(data, list) else data
        
        # Extract parameters
        prompt = inputs_dict.get("prompt", "")
        if isinstance(prompt, bytes):
            prompt = prompt.decode('utf-8')
        
        max_tokens = int(inputs_dict.get("max_tokens", 512))
        temperature = float(inputs_dict.get("temperature", 0.7))
        top_p = float(inputs_dict.get("top_p", 0.9))
        
        # Format prompt for Mistral
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1
        )
        
        # Generate with vLLM
        outputs = llm_engine.generate([formatted_prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        # Format response in Triton format
        response = TritonInferResponse(
            model_name=model_name,
            model_version="1",
            outputs=[
                {
                    "name": "generated_text",
                    "datatype": "BYTES",
                    "shape": [1],
                    "data": [generated_text.encode('utf-8')]
                }
            ]
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.get("/v2/models")
async def list_models():
    """List available models in Triton format"""
    return {
        "models": [
            {
                "name": "vllm_mistral",
                "version": "1",
                "state": "READY"
            }
        ]
    }

@app.get("/v2")
async def server_metadata():
    """Server metadata in Triton format"""
    return {
        "name": "triton-vllm-server",
        "version": "1.0.0",
        "extensions": ["classification", "sequence"]
    }

# Legacy endpoints for compatibility
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Triton-compatible vLLM Server with AWS DLC",
        "model": MODEL_NAME,
        "status": "running",
        "backend": "vllm",
        "triton_compatible": True
    }

if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    
    logger.info(f"Starting Triton-compatible server on {host}:{port}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Max model length: {MAX_MODEL_LEN}")
    logger.info(f"GPU memory utilization: {GPU_MEMORY_UTILIZATION}")
    logger.info(f"Tensor parallel size: {TENSOR_PARALLEL_SIZE}")
    
    uvicorn.run(
        "triton_server_wrapper:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info"
    )
