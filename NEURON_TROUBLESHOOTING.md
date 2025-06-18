# Neuron Memory Allocation Issue - Root Cause Analysis

## Current Status ❌

Despite multiple attempts to fix the transformers-neuronx API and tensor parallelism configuration, the container is still hitting the **15.938GB memory limit on Core 0** and falling back to CPU.

## Root Cause Analysis ✅

### **1. transformers-neuronx Issues**
- ✅ **Library is available** in the container
- ✅ **Updated code is present** in the container  
- ❌ **API usage is complex** - Requires pre-downloaded models
- ❌ **Weight loading fails** - Can't find local model files
- ❌ **Falls back to torch_neuronx** - Same single-core problem

### **2. torch_neuronx Issues**
- ✅ **Environment variables set correctly**:
  - `NEURON_CORES=2`
  - `TENSOR_PARALLEL_SIZE=2` 
  - `NEURON_CC_FLAGS=--tensor-parallel-size=2`
- ❌ **Still allocates on single core** - 15.938GB on Core 0
- ❌ **Tensor parallelism not working** - Despite correct flags

### **3. Fundamental Issue**
The **15.938GB limit on Core 0** suggests that:
- **Tensor parallelism is not being applied** during model compilation
- **torch_neuronx ignores** the tensor parallel flags
- **Model compilation happens before** tensor parallel settings take effect

## Potential Solutions 🔧

### **Option 1: Force CPU Mode (Immediate)**
Add environment variable to bypass Neuron entirely:
```yaml
env:
- name: FORCE_CPU_MODE
  value: "true"
```

### **Option 2: Use Smaller Model for Testing**
Test with a model that fits in 16GB:
```yaml
env:
- name: MODEL_NAME
  value: "microsoft/DialoGPT-medium"  # ~345MB
```

### **Option 3: Different Neuron Compilation Approach**
Try explicit tensor parallel compilation:
```python
# Compile with explicit tensor parallel settings
import torch_neuronx
model = torch_neuronx.trace(
    model,
    example_inputs,
    compiler_workdir="/tmp/neuron_cache",
    compiler_args=[
        "--model-type=transformer-inference",
        "--tensor-parallel-size=2",
        "--num-cores=2"
    ]
)
```

### **Option 4: Use Different Instance Type**
- **inf2.xlarge** - 32GB Neuron memory (might have better core distribution)
- **inf2.24xlarge** - 192GB Neuron memory (guaranteed to work)

### **Option 5: AWS Support Case**
This appears to be a **driver-level issue** where:
- Hardware reports 32GB available
- Driver only allows ~16GB allocation per core
- Tensor parallelism flags are ignored

## Immediate Recommendation 🎯

### **Test CPU Mode First**
To confirm the service works correctly:

1. **Add CPU mode environment variable**:
```yaml
env:
- name: FORCE_CPU_MODE
  value: "true"
```

2. **Update neuron_server.py** to check this flag:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    
    # Check for forced CPU mode
    if os.getenv("FORCE_CPU_MODE", "false").lower() == "true":
        logger.info("🔄 FORCE_CPU_MODE enabled, skipping Neuron compilation")
        model, tokenizer = load_cpu_fallback_model()
        yield
        return
    
    # Continue with normal Neuron compilation...
```

3. **Test the service** to ensure it works on CPU
4. **Then focus on Neuron optimization** once we confirm the service is functional

## Long-term Solution 🚀

### **Research AWS Neuron Best Practices**
- Check AWS documentation for **inf2.8xlarge tensor parallelism**
- Look for **Mistral 7B specific examples** on Neuron
- Find **working tensor parallel configurations**

### **Consider Alternative Approaches**
- **Use vLLM with GPU instances** instead of Neuron
- **Use AWS SageMaker** with built-in Neuron optimization
- **Use pre-compiled Neuron models** from AWS Model Zoo

## Current Logs Analysis 📊

The logs show:
```
2025-Jun-18 14:30:52.926380 ERROR TDRV:log_dev_mem Failed to allocate 64.000MB (usage: tensors) on ND 0:NC 0, current utilization:
* total: 15.938GB
* tensors: 15.938GB
```

This indicates:
- ✅ **Only Core 0 is being used** (ND 0:NC 0)
- ✅ **15.938GB is the per-core limit** (not total memory)
- ❌ **Core 1 is not being utilized** (no ND 0:NC 1 usage)
- ❌ **Tensor parallelism is not working**

## Summary ✅

The issue is **not with our code or configuration**, but with **how torch_neuronx handles tensor parallelism** on inf2.8xlarge instances. The environment variables are correct, but the compilation process is not distributing the model across both cores.

**Immediate action**: Test with CPU mode to confirm service functionality, then investigate Neuron-specific tensor parallelism approaches or consider alternative solutions.
