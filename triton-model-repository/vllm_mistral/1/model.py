import json
import numpy as np
import triton_python_backend_utils as pb_utils
from vllm import LLM, SamplingParams
import torch

class TritonPythonModel:
    """Triton Python backend model for vLLM Mistral 7B"""
    
    def initialize(self, args):
        """Initialize the vLLM model"""
        self.model_config = model_config = json.loads(args['model_config'])
        
        # Get model parameters from environment or config
        model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        max_model_len = 32768
        gpu_memory_utilization = 0.9
        tensor_parallel_size = 1
        
        # Initialize vLLM engine
        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
            dtype="auto",
            enforce_eager=False
        )
        
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
            # Get input tensors
            prompt_tensor = pb_utils.get_input_tensor_by_name(request, "prompt")
            max_tokens_tensor = pb_utils.get_input_tensor_by_name(request, "max_tokens")
            temperature_tensor = pb_utils.get_input_tensor_by_name(request, "temperature")
            top_p_tensor = pb_utils.get_input_tensor_by_name(request, "top_p")
            
            # Extract values
            prompt = prompt_tensor.as_numpy()[0].decode('utf-8')
            max_tokens = int(max_tokens_tensor.as_numpy()[0]) if max_tokens_tensor else 512
            temperature = float(temperature_tensor.as_numpy()[0]) if temperature_tensor else 0.7
            top_p = float(top_p_tensor.as_numpy()[0]) if top_p_tensor else 0.9
            
            # Format prompt for Mistral Instruct
            formatted_prompt = f"<s>[INST] {prompt} [/INST]"
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=1.1
            )
            
            try:
                # Generate response
                outputs = self.llm.generate([formatted_prompt], sampling_params)
                generated_text = outputs[0].outputs[0].text
                
                # Create output tensor
                output_tensor = pb_utils.Tensor(
                    "generated_text",
                    np.array([generated_text.encode('utf-8')], dtype=object)
                )
                
                response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                
            except Exception as e:
                error_msg = f"Generation failed: {str(e)}"
                response = pb_utils.InferenceResponse(
                    error=pb_utils.TritonError(error_msg)
                )
            
            responses.append(response)
        
        return responses
    
    def finalize(self):
        """Clean up resources"""
        if hasattr(self, 'llm'):
            del self.llm
        torch.cuda.empty_cache()
