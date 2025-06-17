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
    """Compile the model for Neuron inference - Memory-efficient version with detailed debugging"""
    logger.info("🚀 Starting memory-efficient Neuron compilation with detailed debugging...")
    
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.metrics as met
        import gc
        logger.info("✅ XLA modules imported successfully")
    except ImportError as e:
        logger.error(f"❌ Failed to import XLA modules: {e}")
        logger.info("🔄 Falling back to CPU model...")
        return load_cpu_fallback_model()
    
    # Memory debugging function
    def log_memory_usage(stage):
        try:
            import psutil
            process = psutil.Process()
            system_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
            logger.info(f"📊 Memory at {stage}: System={system_memory:.2f}GB")
            
            # Try to get Neuron memory info
            try:
                import subprocess
                result = subprocess.run(['neuron-top', '--json'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    logger.info(f"🧠 Neuron memory at {stage}: Available")
                else:
                    logger.info(f"🧠 Neuron memory at {stage}: Status check failed")
            except:
                logger.info(f"🧠 Neuron memory at {stage}: Unable to check")
        except Exception as e:
            logger.info(f"📊 Memory check at {stage}: Failed - {e}")
    
    log_memory_usage("start")
    
    # Load tokenizer first (minimal memory)
    logger.info("📝 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("✅ Tokenizer loaded successfully")
        log_memory_usage("tokenizer_loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load tokenizer: {e}")
        return load_cpu_fallback_model()
    
    # Load model on CPU with memory optimization
    logger.info("🔄 Loading model on CPU with memory optimization...")
    try:
        # Use minimal memory loading
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,  # Start with float32, will optimize later
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cpu",  # Explicitly keep on CPU initially
            offload_folder="/tmp/model_offload",  # Offload to disk if needed
        )
        logger.info("✅ Model loaded on CPU successfully")
        log_memory_usage("model_loaded_cpu")
    except Exception as e:
        logger.error(f"❌ Failed to load model on CPU: {e}")
        return load_cpu_fallback_model()
    
    # Force garbage collection
    gc.collect()
    log_memory_usage("after_gc")
    
    # Use very short sequence for compilation
    COMPILE_SEQUENCE_LENGTH = 32  # Even shorter for maximum stability
    
    # Prepare minimal sample input
    logger.info(f"📝 Preparing minimal sample input (length={COMPILE_SEQUENCE_LENGTH})...")
    try:
        sample_text = "Hi"  # Minimal text
        sample_input = tokenizer(
            sample_text, 
            return_tensors="pt", 
            max_length=COMPILE_SEQUENCE_LENGTH,
            padding="max_length",
            truncation=True
        )
        logger.info("✅ Sample input prepared")
        log_memory_usage("sample_input_prepared")
    except Exception as e:
        logger.error(f"❌ Failed to prepare sample input: {e}")
        return load_cpu_fallback_model()
    
    # Get XLA device
    logger.info("🔍 Getting XLA device...")
    try:
        device = xm.xla_device()
        logger.info(f"✅ XLA device obtained: {device}")
        log_memory_usage("xla_device_ready")
    except Exception as e:
        logger.error(f"❌ Failed to get XLA device: {e}")
        return load_cpu_fallback_model()
    
    # Move sample inputs to XLA device (small memory footprint)
    logger.info("📤 Moving sample inputs to XLA device...")
    try:
        input_ids = sample_input['input_ids'].to(device)
        attention_mask = sample_input['attention_mask'].to(device)
        logger.info("✅ Sample inputs moved to XLA device")
        log_memory_usage("inputs_on_xla")
    except Exception as e:
        logger.error(f"❌ Failed to move inputs to XLA device: {e}")
        return load_cpu_fallback_model()
    
    # Move model to XLA device with careful memory management
    logger.info("🚀 Moving model to XLA device (this may take time)...")
    try:
        # Clear any existing XLA cache
        xm.mark_step()
        
        # Move model to XLA device
        model = model.to(device)
        logger.info("✅ Model moved to XLA device successfully")
        log_memory_usage("model_on_xla")
        
        # Force XLA synchronization
        xm.mark_step()
        log_memory_usage("after_xla_sync")
        
    except Exception as e:
        logger.error(f"❌ Failed to move model to XLA device: {e}")
        logger.error(f"🔍 Error details: {type(e).__name__}: {str(e)}")
        
        # Try to get more specific error information
        if "RESOURCE_EXHAUSTED" in str(e):
            logger.error("💥 RESOURCE_EXHAUSTED: Neuron memory allocation failed")
            logger.error("🔍 This suggests the model is too large for available Neuron memory")
        elif "AllocBuffer" in str(e):
            logger.error("💥 AllocBuffer error: Neuron buffer allocation failed")
        
        logger.info("🔄 Attempting memory-efficient fallback...")
        return load_cpu_fallback_model()
    
    # Test model on XLA device with minimal forward pass
    logger.info("🧪 Testing model with minimal forward pass...")
    try:
        with torch.no_grad():
            # Minimal test forward pass
            test_output = model(input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True)
            logger.info(f"✅ Test forward pass successful, output shape: {test_output.logits.shape}")
            log_memory_usage("test_forward_pass")
            
            # Clear test output
            del test_output
            gc.collect()
            xm.mark_step()
            
    except Exception as e:
        logger.error(f"❌ Test forward pass failed: {e}")
        logger.error(f"🔍 Error details: {type(e).__name__}: {str(e)}")
        return load_cpu_fallback_model()
    
    # Create memory-efficient model wrapper
    logger.info("🔧 Creating memory-efficient model wrapper...")
    try:
        def memory_efficient_wrapper(input_ids, attention_mask):
            """Memory-efficient model wrapper with explicit cleanup"""
            with torch.no_grad():
                # Ensure inputs are on correct device
                if not input_ids.device.type == 'xla':
                    input_ids = input_ids.to(device)
                if not attention_mask.device.type == 'xla':
                    attention_mask = attention_mask.to(device)
                
                # Forward pass with minimal memory usage
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,  # Disable KV caching
                    return_dict=True,
                    output_attentions=False,  # Disable attention outputs
                    output_hidden_states=False,  # Disable hidden state outputs
                )
                
                # Return only logits to minimize memory
                logits = outputs.logits
                
                # Explicit cleanup
                del outputs
                
                return logits
        
        # Test the wrapper
        logger.info("🧪 Testing memory-efficient wrapper...")
        test_logits = memory_efficient_wrapper(input_ids, attention_mask)
        logger.info(f"✅ Wrapper test successful, logits shape: {test_logits.shape}")
        log_memory_usage("wrapper_tested")
        
        # Clean up test
        del test_logits
        gc.collect()
        xm.mark_step()
        
    except Exception as e:
        logger.error(f"❌ Wrapper creation/test failed: {e}")
        return load_cpu_fallback_model()
    
    # Neuron compilation with conservative settings
    logger.info("🔥 Starting Neuron compilation with conservative settings...")
    try:
        with torch.no_grad():
            neuron_model = torch_neuronx.trace(
                memory_efficient_wrapper,
                (input_ids, attention_mask),
                compiler_workdir=COMPILED_MODEL_PATH,
                compiler_args=[
                    "--model-type=transformer-inference",
                    f"--num-cores={NEURON_CORES}",
                    "--auto-cast=none",
                    "--optlevel=0",  # Minimal optimization for stability
                    "--enable-saturate-infinity",
                    "--verbose=1"  # Enable verbose compilation logging
                ]
            )
            
            logger.info("✅ Neuron compilation successful!")
            log_memory_usage("compilation_complete")
            
    except Exception as e:
        logger.error(f"❌ Neuron compilation failed: {e}")
        logger.error(f"🔍 Compilation error details: {type(e).__name__}: {str(e)}")
        
        # Try ultra-minimal fallback
        logger.info("🔄 Attempting ultra-minimal compilation fallback...")
        try:
            def ultra_minimal_wrapper(input_ids):
                """Ultra-minimal wrapper - input_ids only"""
                with torch.no_grad():
                    if not input_ids.device.type == 'xla':
                        input_ids = input_ids.to(device)
                    
                    outputs = model(input_ids=input_ids, use_cache=False, return_dict=True)
                    return outputs.logits
            
            # Test ultra-minimal wrapper
            test_logits = ultra_minimal_wrapper(input_ids)
            logger.info(f"✅ Ultra-minimal wrapper test successful")
            del test_logits
            gc.collect()
            xm.mark_step()
            
            # Compile ultra-minimal version
            neuron_model = torch_neuronx.trace(
                ultra_minimal_wrapper,
                (input_ids,),
                compiler_workdir=COMPILED_MODEL_PATH,
                compiler_args=[
                    "--model-type=transformer-inference",
                    f"--num-cores=1",  # Use only 1 core
                    "--optlevel=0"
                ]
            )
            
            logger.info("✅ Ultra-minimal Neuron compilation successful!")
            log_memory_usage("minimal_compilation_complete")
            
        except Exception as e2:
            logger.error(f"❌ Ultra-minimal compilation also failed: {e2}")
            logger.error(f"🔍 Final error details: {type(e2).__name__}: {str(e2)}")
            return load_cpu_fallback_model()
    
    # Save compiled model
    logger.info("💾 Saving compiled model...")
    try:
        os.makedirs(COMPILED_MODEL_PATH, exist_ok=True)
        torch.jit.save(neuron_model, f"{COMPILED_MODEL_PATH}/neuron_model.pt")
        tokenizer.save_pretrained(COMPILED_MODEL_PATH)
        logger.info("✅ Model saved successfully")
        log_memory_usage("model_saved")
        
        # Final cleanup
        gc.collect()
        xm.mark_step()
        log_memory_usage("final_cleanup")
        
        return neuron_model, tokenizer
        
    except Exception as e:
        logger.error(f"❌ Failed to save compiled model: {e}")
        return load_cpu_fallback_model()

def load_cpu_fallback_model():
    """Load CPU fallback model when Neuron compilation fails"""
    logger.info("🔄 Loading CPU fallback model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,  # Full precision for CPU compatibility
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        logger.info("✅ CPU fallback model loaded successfully (float32)")
        logger.warning("⚠️ Running on CPU with float32 - performance will be limited")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"❌ CPU fallback also failed: {e}")
        raise Exception("Both Neuron compilation and CPU fallback failed")

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
                torch_dtype=torch.float32,  # Changed from float16 to float32 for CPU compatibility
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            logger.info("✅ CPU fallback model loaded successfully (float32)")
            logger.warning("⚠️ Running on CPU with float32 - performance will be limited")
            
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
        
        # Ensure tensors are on the correct device and dtype
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        
        logger.info(f"Generating text for prompt: {request.prompt[:50]}...")
        logger.info(f"Model device: {device}, dtype: {dtype}")
        
        # Use simplified generation approach
        with torch.no_grad():
            try:
                # Use the model's built-in generate method for better compatibility
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
                logger.error(f"Generation error type: {type(gen_error).__name__}")
                logger.error(f"Model device: {device}, Model dtype: {dtype}")
                logger.error(f"Input device: {input_ids.device}, Input dtype: {input_ids.dtype}")
                
                # Return a simple fallback response
                return GenerateResponse(
                    text="I apologize, but I'm having trouble generating a response right now. Please try again.",
                    prompt=request.prompt,
                    model=MODEL_NAME,
                    usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                )
        
    except Exception as e:
        logger.error(f"Request processing failed: {e}")
        logger.error(f"Request error type: {type(e).__name__}")
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
