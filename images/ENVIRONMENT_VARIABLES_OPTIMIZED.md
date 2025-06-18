# Environment Variables Optimized for Tensor Parallelism

## Changes Applied ✅

Both neuron-dlc and neuron-inferentia deployments have been updated with consistent, optimized environment variables for proper tensor parallelism across both Neuron cores.

## Key Changes Made ✅

### **1. Added Tensor Parallelism Support**
```yaml
- name: TENSOR_PARALLEL_SIZE
  value: "2"  # Split model across both 16GB cores
```

### **2. Updated Compiler Flags**
```yaml
- name: NEURON_CC_FLAGS
  value: "--model-type=transformer-inference --optlevel=1 --tensor-parallel-size=2"
```

### **3. Increased Context Length**
```yaml
- name: MAX_LENGTH
  value: "4096"  # Increased from 2048 (neuron-dlc) for better performance
```

### **4. Removed Conflicting/Unnecessary Variables**

#### **Removed from neuron-dlc**:
- ❌ `SEQUENCE_LENGTH` - Redundant with MAX_LENGTH
- ❌ `NEURON_RT_EXEC_TIMEOUT` - Use defaults
- ❌ `NEURON_RT_LOAD_TIMEOUT` - Use defaults  
- ❌ `AWS_DEFAULT_REGION` - Not needed for model loading

#### **Removed from neuron-inferentia**:
- ❌ `SEQUENCE_LENGTH` - Redundant with MAX_LENGTH
- ❌ Duplicate `TENSOR_PARALLEL_SIZE` - Consolidated to single entry

### **5. Updated Optimization Level**
```yaml
- name: NEURON_CC_FLAGS
  value: "--model-type=transformer-inference --optlevel=1 --tensor-parallel-size=2"
```
- Changed from `--optlevel=2` to `--optlevel=1` for stability
- Added `--tensor-parallel-size=2` for proper distribution

## Final Environment Variables ✅

### **Both neuron-dlc and neuron-inferentia now have identical env vars**:

```yaml
env:
- name: MODEL_NAME
  value: "mistralai/Mistral-7B-Instruct-v0.3"
- name: NEURON_CORES
  value: "2"                    # Use both cores
- name: TENSOR_PARALLEL_SIZE
  value: "2"                    # Split model across cores
- name: MAX_LENGTH
  value: "4096"                 # Longer context support
- name: BATCH_SIZE
  value: "1"                    # Single batch for stability
- name: COMPILED_MODEL_PATH
  value: "/tmp/neuron_compiled_model"
- name: HOST
  value: "0.0.0.0"
- name: PORT
  value: "8000"
- name: NEURON_RT_NUM_CORES
  value: "2"                    # Runtime uses both cores
- name: NEURON_CC_FLAGS
  value: "--model-type=transformer-inference --optlevel=1 --tensor-parallel-size=2"
- name: HF_HOME
  value: "/tmp/hf_home"         # Cache location
- name: TOKENIZERS_PARALLELISM
  value: "false"                # Prevent tokenizer conflicts
- name: HF_TOKEN
  valueFrom:
    secretKeyRef:
      name: hugging-face-token
      key: token
```

## Expected Benefits ✅

### **1. Memory Distribution**
- ✅ **Model split across 2 cores** - Each core gets ~8GB instead of 16GB
- ✅ **Access to full 32GB** - 16GB × 2 cores
- ✅ **No single-core memory limit** - Avoids 15.938GB bottleneck

### **2. Performance Optimization**
- ✅ **Parallel processing** across both Neuron cores
- ✅ **Better throughput** with distributed computation
- ✅ **Longer context support** (4096 tokens)

### **3. Stability Improvements**
- ✅ **Lower optimization level** (optlevel=1) for reliability
- ✅ **Consistent configuration** across both images
- ✅ **Removed conflicting variables** that could cause issues

### **4. transformers-neuronx Compatibility**
- ✅ **tp_degree=2** will be read from TENSOR_PARALLEL_SIZE
- ✅ **Proper core distribution** for optimized model loading
- ✅ **Compatible with both torch_neuronx and transformers-neuronx**

## Expected Behavior ✅

### **With Tensor Parallelism**:
```
🚀 Loading Mistral model with transformers-neuronx optimization...
🔧 Initializing optimized Mistral model for Neuron...
   - Tensor parallel degree: 2
   - Using both Neuron cores
   - Memory per core: ~8GB (within 16GB limit)
✅ Optimized Neuron model loaded successfully
```

### **Memory Usage**:
- **Core 0**: ~8GB (half of model)
- **Core 1**: ~8GB (half of model)
- **Total**: ~16GB across both cores
- **Available**: 32GB total (16GB per core)
- **Utilization**: ~50% per core (well within limits)

## Testing Instructions

### **1. Deploy Updated Configuration**:
```bash
# Deploy neuron-dlc
kubectl apply -f images/neuron-dlc/kubernetes-deployment.yaml

# OR deploy neuron-inferentia  
kubectl apply -f images/neuron-inferentia/kubernetes-deployment.yaml
```

### **2. Monitor Logs for Tensor Parallelism**:
```bash
kubectl logs -l app=neuron-mistral-7b -f
```

### **3. Expected Success Logs**:
```
✅ transformers-neuronx available - using optimized Mistral implementation
🔧 Initializing optimized Mistral model for Neuron...
📊 Model configuration:
   - Tensor parallel degree: 2
   - Context length: 4096
✅ Optimized Neuron model loaded successfully
```

### **4. Check Memory Distribution**:
```bash
kubectl exec -it <pod-name> -- neuron-top
# Should show usage distributed across both cores
```

### **5. Test Generation**:
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello, I am",
    "max_tokens": 100
  }'
```

## Summary ✅

The environment variables have been optimized to:

1. ✅ **Enable tensor parallelism** across both 16GB Neuron cores
2. ✅ **Remove conflicting variables** that could cause issues
3. ✅ **Standardize configuration** across both images
4. ✅ **Support longer contexts** (4096 tokens)
5. ✅ **Ensure compatibility** with transformers-neuronx optimization

**🎉 This should resolve the 15.938GB memory limit issue by properly distributing the model across both Neuron cores, giving access to the full 32GB of Neuron memory!**
