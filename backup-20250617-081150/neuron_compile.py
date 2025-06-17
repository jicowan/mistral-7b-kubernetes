#!/usr/bin/env python3
"""
Standalone script to compile Mistral 7B for AWS Neuron
This can be run separately to pre-compile the model
"""

import os
import torch
import torch_neuronx
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compile_mistral_for_neuron(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
    output_dir: str = "/tmp/neuron_compiled_model",
    neuron_cores: int = 2,
    sequence_length: int = 2048,
    batch_size: int = 1
):
    """Compile Mistral 7B model for Neuron inference"""
    
    logger.info(f"Starting compilation of {model_name}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Neuron cores: {neuron_cores}")
    logger.info(f"Sequence length: {sequence_length}")
    logger.info(f"Batch size: {batch_size}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Neuron requires float32
        low_cpu_mem_usage=True
    )
    
    # Set model to evaluation mode
    model.eval()
    
    # Prepare sample inputs for tracing
    logger.info("Preparing sample inputs...")
    sample_text = "Hello, this is a sample input for model compilation."
    sample_inputs = tokenizer(
        sample_text,
        return_tensors="pt",
        max_length=sequence_length,
        padding="max_length",
        truncation=True
    )
    
    input_ids = sample_inputs['input_ids']
    attention_mask = sample_inputs['attention_mask']
    
    logger.info(f"Input shape: {input_ids.shape}")
    
    # Trace the model
    logger.info("Tracing model...")
    with torch.no_grad():
        # Create a wrapper function for tracing
        def model_wrapper(input_ids, attention_mask):
            return model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Trace the model
        traced_model = torch.jit.trace(
            model_wrapper,
            (input_ids, attention_mask),
            strict=False
        )
    
    # Compile for Neuron
    logger.info("Compiling for Neuron... This may take 10-30 minutes.")
    
    compiler_args = [
        "--model-type=transformer",
        f"--num-cores={neuron_cores}",
        "--auto-cast=none",
        "--optlevel=2",
        f"--enable-saturate-infinity",
        f"--enable-mixed-precision-accumulation"
    ]
    
    try:
        neuron_model = torch_neuronx.trace(
            traced_model,
            (input_ids, attention_mask),
            compiler_workdir=f"{output_dir}/workdir",
            compiler_args=compiler_args
        )
        
        # Save the compiled model
        logger.info("Saving compiled model...")
        torch.jit.save(neuron_model, f"{output_dir}/neuron_model.pt")
        
        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        
        # Save compilation metadata
        metadata = {
            "model_name": model_name,
            "neuron_cores": neuron_cores,
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "compiler_args": compiler_args
        }
        
        import json
        with open(f"{output_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("✅ Model compilation completed successfully!")
        logger.info(f"Compiled model saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Compilation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Compile Mistral 7B for AWS Neuron")
    parser.add_argument("--model-name", default="mistralai/Mistral-7B-Instruct-v0.3",
                       help="Hugging Face model name")
    parser.add_argument("--output-dir", default="/tmp/neuron_compiled_model",
                       help="Output directory for compiled model")
    parser.add_argument("--neuron-cores", type=int, default=2,
                       help="Number of Neuron cores to use")
    parser.add_argument("--sequence-length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for compilation")
    
    args = parser.parse_args()
    
    success = compile_mistral_for_neuron(
        model_name=args.model_name,
        output_dir=args.output_dir,
        neuron_cores=args.neuron_cores,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size
    )
    
    if success:
        print("\n🎉 Compilation completed successfully!")
        print(f"You can now use the compiled model from: {args.output_dir}")
    else:
        print("\n❌ Compilation failed. Check the logs above.")
        exit(1)

if __name__ == "__main__":
    main()
