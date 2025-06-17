# Update K8s Images Script - Updated for New Structure

## Script Updated ✅

The `update-k8s-images.sh` script has been completely rewritten to work with the new organized project structure where each image has its own directory with a `kubernetes-deployment.yaml` file.

## Key Changes Made

### **Before (Old Structure)**:
```bash
# Fixed file paths
kubernetes-deployment.yaml
kubernetes-deployment-triton.yaml  
kubernetes-deployment-neuron.yaml
aws-dlc/kubernetes-deployment-vllm-dlc.yaml
aws-dlc/kubernetes-deployment-triton-dlc.yaml
aws-dlc/kubernetes-deployment-neuron-dlc.yaml
```

### **After (New Structure)**:
```bash
# Organized by image type
images/vllm-gpu/kubernetes-deployment.yaml
images/triton-gpu/kubernetes-deployment.yaml
images/neuron-inferentia/kubernetes-deployment.yaml
images/vllm-dlc/kubernetes-deployment.yaml
images/triton-dlc/kubernetes-deployment.yaml
images/neuron-dlc/kubernetes-deployment.yaml
```

## New Features ✅

### **1. Selective Updates**
```bash
# Update all deployments
./update-k8s-images.sh all

# Update specific deployment
./update-k8s-images.sh vllm-gpu
./update-k8s-images.sh triton-dlc
```

### **2. Flexible Registry Support**
```bash
# Local images (no ECR)
./update-k8s-images.sh vllm-gpu

# With ECR registry
./update-k8s-images.sh vllm-gpu us-west-2 123456789012
```

### **3. Automatic AWS Account Detection**
```bash
# Auto-detect AWS account
./update-k8s-images.sh vllm-gpu us-west-2

# Manual account specification
./update-k8s-images.sh vllm-gpu us-west-2 123456789012
```

### **4. Better Error Handling**
- ✅ Validates image directory exists
- ✅ Checks for deployment file existence
- ✅ Creates backups before modification
- ✅ Provides clear error messages

## Usage Examples

### **Update All Deployments**
```bash
./update-k8s-images.sh all us-west-2
```
**Result**: Updates all 6 deployment files with ECR URIs

### **Update Specific Image**
```bash
./update-k8s-images.sh triton-dlc us-west-2
```
**Result**: Updates only the Triton DLC deployment

### **Local Development**
```bash
./update-k8s-images.sh vllm-gpu
```
**Result**: Uses local image names (no ECR registry)

### **Custom AWS Account**
```bash
./update-k8s-images.sh all us-east-1 123456789012
```
**Result**: Uses specific AWS account and region

## Image Mapping ✅

| Directory | ECR Image Name | Deployment File |
|-----------|----------------|-----------------|
| `vllm-gpu` | `vllm-mistral-7b` | `images/vllm-gpu/kubernetes-deployment.yaml` |
| `triton-gpu` | `triton-vllm-mistral-7b` | `images/triton-gpu/kubernetes-deployment.yaml` |
| `neuron-inferentia` | `neuron-mistral-7b` | `images/neuron-inferentia/kubernetes-deployment.yaml` |
| `vllm-dlc` | `vllm-mistral-7b-dlc` | `images/vllm-dlc/kubernetes-deployment.yaml` |
| `triton-dlc` | `triton-mistral-7b-dlc` | `images/triton-dlc/kubernetes-deployment.yaml` |
| `neuron-dlc` | `neuron-mistral-7b-dlc` | `images/neuron-dlc/kubernetes-deployment.yaml` |

## Script Output Example

```bash
$ ./update-k8s-images.sh triton-dlc us-west-2

🔄 Updating Kubernetes deployment files with ECR image URIs
==========================================================
Target: triton-dlc
AWS Region: us-west-2
AWS Account: 123456789012
ECR Registry: 123456789012.dkr.ecr.us-west-2.amazonaws.com

1. Checking prerequisites...
✅ Prerequisites check passed

2. Updating deployment files...
Updating single deployment: triton-dlc
   Updating triton-dlc deployment...
     File: images/triton-dlc/kubernetes-deployment.yaml
     Image: triton-mistral-7b-dlc
     ✅ Updated to: 123456789012.dkr.ecr.us-west-2.amazonaws.com/triton-mistral-7b-dlc:latest

3. Updated Image URIs:
=====================
   triton-dlc: 123456789012.dkr.ecr.us-west-2.amazonaws.com/triton-mistral-7b-dlc:latest

4. Deployment Commands:
======================
Deploy triton-dlc:
   kubectl apply -f images/triton-dlc/kubernetes-deployment.yaml

✅ Kubernetes deployment files updated!
```

## Safety Features ✅

### **Backup Creation**
- ✅ Creates `.bak` files before modification
- ✅ Preserves original deployment files
- ✅ Easy rollback if needed

### **Validation**
- ✅ Checks if deployment files exist
- ✅ Validates image directory names
- ✅ Verifies AWS CLI availability when needed

### **Flexible Operation**
- ✅ Works with or without AWS credentials
- ✅ Handles missing deployment files gracefully
- ✅ Provides clear usage instructions

## Integration with Build Process

### **Complete Workflow**
```bash
# 1. Build images
./build-all-images.sh all us-west-2 123456789012

# 2. Update deployments
./update-k8s-images.sh all us-west-2 123456789012

# 3. Deploy to Kubernetes
kubectl apply -f images/vllm-gpu/kubernetes-deployment.yaml
kubectl apply -f images/triton-dlc/kubernetes-deployment.yaml
```

### **Individual Image Workflow**
```bash
# 1. Build specific image
./build-all-images.sh triton-dlc us-west-2 123456789012

# 2. Update its deployment
./update-k8s-images.sh triton-dlc us-west-2 123456789012

# 3. Deploy
kubectl apply -f images/triton-dlc/kubernetes-deployment.yaml
```

## Benefits of the Update ✅

### **Organization**
- ✅ **Consistent with new structure** - Works with organized image directories
- ✅ **Self-contained** - Each image has its own deployment file
- ✅ **Easy to find** - Deployment files co-located with image files

### **Flexibility**
- ✅ **Selective updates** - Update only what you need
- ✅ **Multiple registries** - Works with any container registry
- ✅ **Local development** - Works without AWS credentials

### **Maintenance**
- ✅ **Easier to maintain** - Clear mapping between images and deployments
- ✅ **Better error handling** - Clear messages when things go wrong
- ✅ **Safer operation** - Backups and validation built-in

### **Developer Experience**
- ✅ **Intuitive usage** - Clear command-line interface
- ✅ **Helpful output** - Shows exactly what was updated
- ✅ **Ready-to-use commands** - Provides kubectl commands to run

🎉 **The update-k8s-images.sh script is now fully compatible with the new organized project structure!**
