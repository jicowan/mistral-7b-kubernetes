#!/usr/bin/env python3
"""
AWS Neuron Server for Mistral 7B Instruct
Optimized for Inferentia 1 and Inferentia 2 chips
"""

import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch_neuronx
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig
import neuronx_distributed as nxd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
NEURON_CORES = int(os.getenv("NEURON_CORES", "2"))  # Number of Neuron cores to use
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "2048"))  # Reduced for Inferentia
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", "2048"))
COMPILED_MODEL_PATH = os.getenv("COMPILED_MODEL_PATH", "/tmp/neuron_compiled_model")

# Global model and tokenizer
model = None
tokenizer = None

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
    neuron_cores: int
    device_type: str

def compile_model_for_neuron():
    """Compile the model for Neuron inference"""
    logger.info("Starting model compilation for Neuron...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in float32 first
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    # Prepare sample input for tracing
    sample_input = tokenizer(
        "Hello, how are you?", 
        return_tensors="pt", 
        max_length=SEQUENCE_LENGTH,
        padding="max_length",
        truncation=True
    )
    
    logger.info("Tracing model for Neuron compilation...")
    
    # Trace the model for Neuron
    with torch.no_grad():
        traced_model = torch.jit.trace(
            model,
            (sample_input['input_ids'], sample_input['attention_mask']),
            strict=False
        )
    
    # Compile for Neuron
    logger.info("Compiling traced model for Neuron...")
    neuron_model = torch_neuronx.trace(
        traced_model,
        (sample_input['input_ids'], sample_input['attention_mask']),
        compiler_workdir=COMPILED_MODEL_PATH,
        compiler_args=[
            "--model-type=transformer",
            f"--num-cores={NEURON_CORES}",
            "--auto-cast=none",
            "--optlevel=2"
        ]
    )
    
    # Save compiled model
    os.makedirs(COMPILED_MODEL_PATH, exist_ok=True)
    torch.jit.save(neuron_model, f"{COMPILED_MODEL_PATH}/neuron_model.pt")
    tokenizer.save_pretrained(COMPILED_MODEL_PATH)
    
    logger.info("Model compilation completed successfully")
    return neuron_model, tokenizer

def load_compiled_model():
    """Load pre-compiled Neuron model"""
    logger.info("Loading compiled Neuron model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(COMPILED_MODEL_PATH)
    
    # Load compiled model
    model = torch.jit.load(f"{COMPILED_MODEL_PATH}/neuron_model.pt")
    
    logger.info("Compiled model loaded successfully")
    return model, tokenizer

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup the Neuron model"""
    global model, tokenizer
    
    logger.info("Starting Neuron model initialization...")
    
    try:
        # Check if compiled model exists
        if os.path.exists(f"{COMPILED_MODEL_PATH}/neuron_model.pt"):
            logger.info("Found pre-compiled model, loading...")
            model, tokenizer = load_compiled_model()
        else:
            logger.info("No pre-compiled model found, compiling...")
            model, tokenizer = compile_model_for_neuron()
        
        logger.info(f"Neuron model initialized successfully with {NEURON_CORES} cores")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize Neuron model: {e}")
        raise
    finally:
        if model:
            logger.info("Cleaning up Neuron model...")
            del model
            del tokenizer

# Create FastAPI app with lifespan
app = FastAPI(
    title="AWS Neuron Mistral 7B Server",
    description="High-performance inference server for Mistral 7B Instruct using AWS Neuron",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    # Detect Inferentia version
    device_type = "inferentia1"
    try:
        import subprocess
        result = subprocess.run(['neuron-ls'], capture_output=True, text=True)
        if 'inf2' in result.stdout.lower():
            device_type = "inferentia2"
    except:
        pass
    
    return HealthResponse(
        status="healthy",
        model=MODEL_NAME,
        neuron_cores=NEURON_CORES,
        device_type=device_type
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """Generate text using the Neuron-compiled Mistral model"""
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    try:
        # Format prompt for Mistral Instruct
        formatted_prompt = f"<s>[INST] {request.prompt} [/INST]"
        
        # Tokenize input
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=SEQUENCE_LENGTH,
            padding="max_length",
            truncation=True
        )
        
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Calculate actual prompt length (excluding padding)
        prompt_length = (input_ids != tokenizer.pad_token_id).sum().item()
        
        # Generate with Neuron model
        with torch.no_grad():
            # For Neuron, we need to handle generation differently
            # This is a simplified approach - you may need to implement
            # custom generation logic for better performance
            
            generated_ids = input_ids.clone()
            
            for _ in range(min(request.max_tokens, MAX_LENGTH - prompt_length)):
                # Get model predictions
                outputs = model(generated_ids, attention_mask)
                
                # Get next token logits
                next_token_logits = outputs.logits[:, -1, :]
                
                # Apply temperature
                if request.temperature > 0:
                    next_token_logits = next_token_logits / request.temperature
                
                # Apply top-k filtering
                if request.top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, request.top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p filtering
                if request.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > request.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Check for stop tokens
                if next_token.item() == tokenizer.eos_token_id:
                    break
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # Update attention mask
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((1, 1), dtype=attention_mask.dtype)
                ], dim=-1)
        
        # Decode generated text
        generated_text = tokenizer.decode(
            generated_ids[0][prompt_length:], 
            skip_special_tokens=True
        )
        
        # Calculate usage statistics
        completion_tokens = generated_ids.shape[1] - prompt_length
        total_tokens = generated_ids.shape[1]
        
        return GenerateResponse(
            text=generated_text,
            prompt=request.prompt,
            model=MODEL_NAME,
            usage={
                "prompt_tokens": prompt_length,
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
                "owned_by": "mistralai",
                "backend": "neuron"
            }
        ]
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AWS Neuron Mistral 7B Server",
        "model": MODEL_NAME,
        "status": "running",
        "backend": "neuron"
    }

if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    
    logger.info(f"Starting Neuron server on {host}:{port}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Neuron cores: {NEURON_CORES}")
    logger.info(f"Max length: {MAX_LENGTH}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    
    uvicorn.run(
        "neuron_server:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info"
    )
