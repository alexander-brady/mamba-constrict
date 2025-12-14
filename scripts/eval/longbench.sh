#!/bin/bash
#SBATCH --job-name=longbench_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --tmp=64G
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpumem:32g
#SBATCH --mail-type=END,FAIL

# Usage: sbatch scripts/longbench.sh [gpu_memory_utilization] [tensor_parallel_size]

GPU_MEMORY_UTILIZATION=${1:-0.95}
TENSOR_PARALLEL_SIZE=${2:-1}

# Source environment
if [ -f "scripts/euler/env.sh" ]; then
    source scripts/euler/env.sh
fi

# Enable verbose vLLM logging
export VLLM_LOGGING_LEVEL=DEBUG

# Set up random port and key to avoid conflicts
PORT=8000
export VLLM_API_KEY="token-$(date +%s)"
export VLLM_URL="http://localhost:$PORT/v1"

echo "Port: $PORT"
echo "URL: $VLLM_URL"

# Get list of models from config
MODEL_NAMES=$(python3 -c 'import json; print(" ".join([k for k in json.load(open("eval/config/model2path.json")).keys() if not k.startswith("_")]))')

for MODEL_NAME in $MODEL_NAMES; do
    MODEL_PATH=$(python3 -c "import json; print(json.load(open('eval/config/model2path.json'))['$MODEL_NAME'])")
    MAX_LEN=$(python3 -c "import json; print(json.load(open('eval/config/model2maxlen.json'))['$MODEL_NAME'])")

    echo "----------------------------------------------------------------"
    echo "Processing Model: $MODEL_NAME"
    echo "Model Path: $MODEL_PATH"
    echo "Max Length: $MAX_LEN"
    echo "----------------------------------------------------------------"

    # Start vLLM
    echo "Starting vLLM server for $MODEL_NAME..."
    vllm serve $MODEL_PATH \
        --api-key $VLLM_API_KEY \
        --port $PORT \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --no-enable-chunked-prefill \
        --max_model_len $MAX_LEN \
        --trust-remote-code 2>&1 &

    VLLM_PID=$!

    # Wait for vLLM to be ready (no time limit - wait until process dies or API responds)
    echo "Waiting for vLLM to be ready..."
    i=0
    while true; do
        i=$((i + 1))
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "vLLM process died unexpectedly!"
            wait $VLLM_PID
            echo "Exit code: $?"
            break
        fi
        if curl -s $VLLM_URL/models > /dev/null 2>&1; then
            echo "vLLM is ready!"
            break
        fi
        echo "Waiting... ($i)"
        sleep 10
    done
    if ! curl -s $VLLM_URL/models > /dev/null 2>&1; then
        echo "vLLM failed to start for $MODEL_NAME."
        kill $VLLM_PID 2>/dev/null || true
        continue
    fi
    echo "vLLM is ready!"

    # Run inference
    pushd eval/LongBench > /dev/null
    python pred.py --model $MODEL_NAME --save_dir ../../results/longbench

    # Export results
    python result.py --results_dir ../../results/longbench
    popd > /dev/null

    # Cleanup
    kill $VLLM_PID
    
    # Wait a bit to ensure port is freed
    sleep 10
done
