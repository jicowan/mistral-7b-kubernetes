#!/usr/bin/env python3
"""
vLLM Server for Mistral 7B Instruct
Optimized for NVIDIA A10G and L4 GPUs
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "32768"))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "true").lower() == "true"

# Global engine variable
engine: Optional[AsyncLLMEngine] = None

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    stop: Optional[List[str]] = None

class GenerateResponse(BaseModel):
    text: str
    prompt: str
    model: str
    usage: Dict[str, int]

class HealthResponse(BaseModel):
    status: str
    model: str
    gpu_memory_utilization: float

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup the vLLM engine"""
    global engine
    
    logger.info("Starting vLLM engine initialization...")
    
    try:
        # Configure engine arguments with optimizations for CUDA 12.9
        engine_args = AsyncEngineArgs(
            model=MODEL_NAME,
            max_model_len=MAX_MODEL_LEN,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            trust_remote_code=TRUST_REMOTE_CODE,
            dtype="auto",  # Let vLLM choose the best dtype
            enforce_eager=False,  # Use CUDA graphs for better performance
            disable_log_stats=False,
            # Enhanced settings for newer CUDA/vLLM versions
            enable_prefix_caching=True,  # Better performance for repeated prompts
            max_num_seqs=256,  # Increased batch size for better throughput
        )
        
        # Create the async engine
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info(f"vLLM engine initialized successfully with model: {MODEL_NAME}")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize vLLM engine: {e}")
        raise
    finally:
        if engine:
            logger.info("Shutting down vLLM engine...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="vLLM Mistral 7B Server",
    description="High-performance inference server for Mistral 7B Instruct using vLLM",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    return HealthResponse(
        status="healthy",
        model=MODEL_NAME,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text using the Mistral model"""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        # Create sampling parameters
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            stop=request.stop or []
        )
        
        # Format prompt for Mistral Instruct
        formatted_prompt = f"<s>[INST] {request.prompt} [/INST]"
        
        # Generate response
        results = []
        async for request_output in engine.generate(
            formatted_prompt, 
            sampling_params, 
            request_id=None
        ):
            results.append(request_output)
        
        if not results:
            raise HTTPException(status_code=500, detail="No output generated")
        
        # Get the final result
        final_output = results[-1]
        generated_text = final_output.outputs[0].text
        
        # Calculate usage statistics
        prompt_tokens = len(final_output.prompt_token_ids)
        completion_tokens = len(final_output.outputs[0].token_ids)
        total_tokens = prompt_tokens + completion_tokens
        
        return GenerateResponse(
            text=generated_text,
            prompt=request.prompt,
            model=MODEL_NAME,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        )
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 1677610602,
                "owned_by": "mistralai"
            }
        ]
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "vLLM Mistral 7B Server",
        "model": MODEL_NAME,
        "status": "running"
    }

if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Max model length: {MAX_MODEL_LEN}")
    logger.info(f"GPU memory utilization: {GPU_MEMORY_UTILIZATION}")
    logger.info(f"Tensor parallel size: {TENSOR_PARALLEL_SIZE}")
    
    uvicorn.run(
        "vllm_server:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info"
    )
