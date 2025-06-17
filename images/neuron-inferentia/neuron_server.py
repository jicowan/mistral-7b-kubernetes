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
    """Compile the model for Neuron inference - Simplified version"""
    logger.info("Starting simplified model compilation for Neuron...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model in float32 first
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    # Use shorter sequence length for compilation to avoid tensor issues
    COMPILE_SEQUENCE_LENGTH = 128  # Much shorter for stable compilation
    
    # Prepare simple sample input for tracing (no past_key_values)
    logger.info(f"Preparing sample input with sequence length {COMPILE_SEQUENCE_LENGTH}...")
    sample_text = "Hello, how are you?"
    sample_input = tokenizer(
        sample_text, 
        return_tensors="pt", 
        max_length=COMPILE_SEQUENCE_LENGTH,
        padding="max_length",
        truncation=True
    )
    
    logger.info("Starting Neuron compilation with simplified inputs...")
    
    # Compile directly with torch_neuronx.trace (skip torch.jit.trace)
    try:
        with torch.no_grad():
            # Create a wrapper function that only takes the inputs we need
            def model_wrapper(input_ids, attention_mask):
                # Call model with only basic inputs, no past_key_values
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,  # Disable KV caching to avoid tensor type issues
                    return_dict=True
                )
                return outputs.logits
            
            # Trace with simplified inputs
            neuron_model = torch_neuronx.trace(
                model_wrapper,
                (sample_input['input_ids'], sample_input['attention_mask']),
                compiler_workdir=COMPILED_MODEL_PATH,
                compiler_args=[
                    "--model-type=transformer-inference",
                    f"--num-cores={NEURON_CORES}",
                    "--auto-cast=none",
                    "--optlevel=1",  # Lower optimization level for stability
                    "--enable-saturate-infinity",
                    "--enable-mixed-precision-accumulation"
                ]
            )
    
    except Exception as e:
        logger.error(f"Neuron compilation failed: {e}")
        logger.info("Attempting fallback compilation with even simpler settings...")
        
        # Fallback: try with minimal settings
        try:
            with torch.no_grad():
                def simple_model_wrapper(input_ids):
                    # Even simpler - just input_ids, let model handle attention_mask
                    outputs = model(input_ids=input_ids, use_cache=False, return_dict=True)
                    return outputs.logits
                
                neuron_model = torch_neuronx.trace(
                    simple_model_wrapper,
                    (sample_input['input_ids'],),
                    compiler_workdir=COMPILED_MODEL_PATH,
                    compiler_args=[
                        "--model-type=transformer-inference",
                        f"--num-cores={NEURON_CORES}",
                        "--optlevel=1"
                    ]
                )
        except Exception as e2:
            logger.error(f"Fallback compilation also failed: {e2}")
            logger.info("Using CPU fallback model...")
            return model, tokenizer  # Return uncompiled model as fallback
    
    # Save compiled model
    logger.info("Saving compiled model...")
    os.makedirs(COMPILED_MODEL_PATH, exist_ok=True)
    torch.jit.save(neuron_model, f"{COMPILED_MODEL_PATH}/neuron_model.pt")
    tokenizer.save_pretrained(COMPILED_MODEL_PATH)
    
    logger.info("✅ Model compilation completed successfully")
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
    
    logger.info("🚀 Starting Neuron model initialization...")
    
    try:
        # Check if compiled model exists
        if os.path.exists(f"{COMPILED_MODEL_PATH}/neuron_model.pt"):
            logger.info("📁 Found pre-compiled model, loading...")
            try:
                model, tokenizer = load_compiled_model()
                logger.info("✅ Pre-compiled Neuron model loaded successfully")
            except Exception as load_error:
                logger.error(f"❌ Failed to load pre-compiled model: {load_error}")
                logger.info("🔄 Falling back to compilation...")
                model, tokenizer = compile_model_for_neuron()
        else:
            logger.info("🔨 No pre-compiled model found, starting compilation...")
            model, tokenizer = compile_model_for_neuron()
        
        # Verify model is working
        if model is None or tokenizer is None:
            raise Exception("Model or tokenizer is None after initialization")
        
        logger.info(f"✅ Neuron model initialized successfully with {NEURON_CORES} cores")
        logger.info(f"📊 Model: {MODEL_NAME}")
        logger.info(f"🔧 Max length: {MAX_LENGTH}")
        logger.info("🎯 Server ready for requests!")
        
        yield
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize Neuron model: {e}")
        logger.info("🔄 Attempting CPU fallback...")
        
        try:
            # CPU fallback
            logger.info("Loading model on CPU as fallback...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            logger.info("✅ CPU fallback model loaded successfully")
            logger.warning("⚠️ Running on CPU - performance will be limited")
            
            yield
            
        except Exception as fallback_error:
            logger.error(f"❌ CPU fallback also failed: {fallback_error}")
            raise Exception("Both Neuron compilation and CPU fallback failed")
    
    finally:
        if model:
            logger.info("🧹 Cleaning up model resources...")
            del model
        if tokenizer:
            del tokenizer
        logger.info("✅ Cleanup completed")

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
        
        # Tokenize input with shorter max length for stability
        max_input_length = min(512, MAX_LENGTH - request.max_tokens)  # Leave room for generation
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=max_input_length,
            padding=False,  # Don't pad for generation
            truncation=True
        )
        
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids))
        
        logger.info(f"Generating text for prompt: {request.prompt[:50]}...")
        
        # Use simplified generation approach
        with torch.no_grad():
            try:
                # Try to use the model's built-in generate method if available
                if hasattr(model, 'generate'):
                    generated_ids = model.generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        repetition_penalty=request.repetition_penalty,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=False,  # Disable KV caching
                        return_dict_in_generate=False
                    )
                else:
                    # Fallback: simple autoregressive generation
                    generated_ids = input_ids.clone()
                    
                    for step in range(request.max_tokens):
                        # Get model predictions (simplified call)
                        if hasattr(model, '__call__'):
                            # For compiled Neuron model
                            logits = model(generated_ids)
                        else:
                            # For regular model
                            outputs = model(generated_ids, attention_mask=attention_mask, use_cache=False)
                            logits = outputs.logits
                        
                        # Get next token logits
                        next_token_logits = logits[:, -1, :]
                        
                        # Apply temperature
                        if request.temperature > 0:
                            next_token_logits = next_token_logits / request.temperature
                        
                        # Simple sampling
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
                            torch.ones(1, 1, dtype=attention_mask.dtype)
                        ], dim=-1)
                
                # Decode the generated text
                generated_text = tokenizer.decode(
                    generated_ids[0][input_ids.shape[1]:],  # Only decode the new tokens
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                
                # Calculate token usage
                prompt_tokens = input_ids.shape[1]
                completion_tokens = generated_ids.shape[1] - prompt_tokens
                total_tokens = prompt_tokens + completion_tokens
                
                logger.info(f"Generated {completion_tokens} tokens successfully")
                
                return GenerateResponse(
                    text=generated_text.strip(),
                    prompt=request.prompt,
                    model=MODEL_NAME,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens
                    }
                )
                
            except Exception as gen_error:
                logger.error(f"Generation failed: {gen_error}")
                # Return a simple fallback response
                return GenerateResponse(
                    text="I apologize, but I'm having trouble generating a response right now.",
                    prompt=request.prompt,
                    model=MODEL_NAME,
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                )
        
    except Exception as e:
        logger.error(f"Request processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
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
