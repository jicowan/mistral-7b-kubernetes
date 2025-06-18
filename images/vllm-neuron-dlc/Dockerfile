# AWS Neuron DLC + vLLM for Mistral 7B on Inferentia2
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference-neuronx:2.1.2-neuronx-py310-sdk2.20.0-ubuntu20.04

# Install vLLM with Neuron support
RUN pip install --no-cache-dir vllm[neuron]==0.6.2

# Set environment variables
ENV MODEL_ID=mistralai/Mistral-7B-Instruct-v0.3
ENV DEVICE=neuron
ENV TENSOR_PARALLEL_SIZE=2
ENV MAX_NUM_SEQS=4
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start vLLM OpenAI API server
CMD python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_ID \
    --device $DEVICE \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --max-num-seqs $MAX_NUM_SEQS \
    --host $HOST \
    --port $PORT
