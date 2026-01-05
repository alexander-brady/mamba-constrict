#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH --time=01:30:00
#SBATCH --job-name=vllm-smoke-test
#SBATCH --output=logs/vllm-smoke-test-%j.out
#SBATCH --error=logs/vllm-smoke-test-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --environment=vllm
#SBATCH --no-requeue
#SBATCH -C thp_never&nvidia_vboost_enabled

echo "Starting vLLM smoke test at $(date)"

# Create logs directory if it doesn't exist
mkdir -p logs

# Test script path
TEST_SCRIPT="$SCRATCH/finetune/scripts/cscs/eval/test_vllm_inference.py"

# Default model - use a small Mamba model for testing
# User can override with MODEL_PATH environment variable
MODEL_PATH="${MODEL_PATH:-state-spaces/mamba-130m-hf}"

echo "Using model: $MODEL_PATH"
echo "Running vLLM inference test..."

python3 "$TEST_SCRIPT" --model "$MODEL_PATH"

echo "Finished vLLM smoke test at $(date)"

