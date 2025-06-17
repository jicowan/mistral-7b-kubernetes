import json
import numpy as np
import triton_python_backend_utils as pb_utils
from vllm import LLM, SamplingParams
import torch
import os

class TritonPythonModel:
    """Triton Python backend model for vLLM Mistral 7B"""
    
    def initialize(self, args):
        """Initialize the vLLM model"""
        self.model_config = model_config = json.loads(args['model_config'])
        
        # Get model parameters from environment or config
        model_name = os.getenv("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
        max_model_len = int(os.getenv("MAX_MODEL_LEN", "32768"))
        gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))
        tensor_parallel_size = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
        
        # Initialize vLLM engine with enhanced settings
        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            dtype="auto",
            enforce_eager=False,
            # Enhanced settings for better performance
            enable_prefix_caching=True,
            max_num_seqs=256,
        )
        
        print(f"Triton vLLM model initialized: {model_name}")
        
        # Get output configuration
        output_config = pb_utils.get_output_config_by_name(
            model_config, "generated_text"
        )
        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_config['data_type']
        )
        
    def execute(self, requests):
        """Execute inference requests"""
        responses = []
        
        for request in requests:
            try:
                # Get input tensors
                prompt_tensor = pb_utils.get_input_tensor_by_name(request, "prompt")
                max_tokens_tensor = pb_utils.get_input_tensor_by_name(request, "max_tokens")
                temperature_tensor = pb_utils.get_input_tensor_by_name(request, "temperature")
                top_p_tensor = pb_utils.get_input_tensor_by_name(request, "top_p")
                
                # Extract values with robust error handling
                prompt = ""
                if prompt_tensor:
                    prompt_raw = prompt_tensor.as_numpy()[0]
                    if isinstance(prompt_raw, bytes):
                        prompt = prompt_raw.decode('utf-8')
                    elif isinstance(prompt_raw, str):
                        prompt = prompt_raw
                    elif hasattr(prompt_raw, 'item'):
                        # Handle numpy scalar
                        prompt_item = prompt_raw.item()
                        if isinstance(prompt_item, bytes):
                            prompt = prompt_item.decode('utf-8')
                        elif isinstance(prompt_item, str):
                            prompt = prompt_item
                        else:
                            prompt = str(prompt_item)
                    else:
                        prompt = str(prompt_raw)
                
                # Extract other parameters safely
                max_tokens = 512
                if max_tokens_tensor:
                    try:
                        max_tokens_raw = max_tokens_tensor.as_numpy()[0]
                        if hasattr(max_tokens_raw, 'item'):
                            max_tokens = int(max_tokens_raw.item())
                        else:
                            max_tokens = int(max_tokens_raw)
                    except (ValueError, TypeError):
                        max_tokens = 512
                
                temperature = 0.7
                if temperature_tensor:
                    try:
                        temp_raw = temperature_tensor.as_numpy()[0]
                        if hasattr(temp_raw, 'item'):
                            temperature = float(temp_raw.item())
                        else:
                            temperature = float(temp_raw)
                    except (ValueError, TypeError):
                        temperature = 0.7
                
                top_p = 0.9
                if top_p_tensor:
                    try:
                        top_p_raw = top_p_tensor.as_numpy()[0]
                        if hasattr(top_p_raw, 'item'):
                            top_p = float(top_p_raw.item())
                        else:
                            top_p = float(top_p_raw)
                    except (ValueError, TypeError):
                        top_p = 0.9
                
                if not prompt:
                    raise ValueError("Empty prompt provided")
                
                print(f"Processing prompt: {prompt[:50]}...")  # Debug log
                
                # Format prompt for Mistral Instruct
                formatted_prompt = f"<s>[INST] {prompt} [/INST]"
                
                # Create sampling parameters
                sampling_params = SamplingParams(
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=1.1,
                    # Additional parameters for better quality
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                )
                
                # Generate response
                outputs = self.llm.generate([formatted_prompt], sampling_params)
                generated_text = outputs[0].outputs[0].text
                
                print(f"Generated response: {generated_text[:50]}...")  # Debug log
                
                # Create output tensor
                output_tensor = pb_utils.Tensor(
                    "generated_text",
                    np.array([generated_text.encode('utf-8')], dtype=object)
                )
                
                response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                
            except Exception as e:
                error_msg = f"Generation failed: {str(e)}"
                print(f"Error in Triton model execution: {error_msg}")
                import traceback
                traceback.print_exc()  # Print full stack trace for debugging
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(error_msg)
                )
            
            responses.append(response)
        
        return responses
    
    def finalize(self):
        """Clean up resources"""
        print("Finalizing Triton vLLM model...")
        if hasattr(self, 'llm'):
            del self.llm
        torch.cuda.empty_cache()
        print("Triton vLLM model finalized")
