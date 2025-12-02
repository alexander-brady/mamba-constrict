#!/usr/bin/env bash
# Usage: ./scripts/eval/babilong.sh [tensor_parallel_size]
# Example: CUDA_VISIBLE_DEVICES=0,1 ./scripts/eval/babilong.sh 2
set -e

# Function to check if the API server is ready
wait_for_server() {
    echo "Waiting for vLLM server to start..."
    while true; do
        if ! kill -0 $VLLM_PID 2>/dev/null; then
            echo "vLLM process failed to start!"
            return 1
        fi
        if curl -s "${VLLM_API_URL}/completions" &>/dev/null; then
            echo "vLLM server is ready!"
            return 0
        fi
        sleep 1
    done
}

# Function to kill the vLLM server
cleanup() {
    echo "Stopping vLLM server..."
    pkill -f "vllm serve" || true
}

# Source environment
if [ -f "scripts/env.sh" ]; then
    source scripts/env.sh
fi

# API configuration
VLLM_API_HOST="${VLLM_API_HOST:-localhost}"
VLLM_API_PORT="${VLLM_API_PORT:-8000}"
VLLM_API_URL="${VLLM_API_URL:-http://${VLLM_API_HOST}:${VLLM_API_PORT}/v1}"

# Tensor parallel size - accepts command line argument or environment variable
TENSOR_PARALLEL_SIZE=${1:-${TP:-1}}

RESULTS_FOLDER="./results/babilong"

# Get list of models from config
MODEL_NAMES=$(python3 -c 'import json; print(" ".join([k for k in json.load(open("eval/config/model2path.json")).keys() if not k.startswith("_")]))')

for MODEL_NAME in $MODEL_NAMES; do
    MODEL_PATH=$(python3 -c "import json; print(json.load(open('eval/config/model2path.json'))['$MODEL_NAME'])")

    echo "================================================================"
    echo "Processing Model: $MODEL_NAME"
    echo "Model Path: $MODEL_PATH"
    echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
    echo "================================================================"

    # Start the vLLM server in the background
    echo "Starting vLLM server for $MODEL_NAME..."
    vllm serve "$MODEL_PATH" --enable-chunked-prefill=False --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --served-model-name "$MODEL_NAME" --host "${VLLM_API_HOST}" --port "${VLLM_API_PORT}" --disable-log-requests &

    VLLM_PID=$!
    echo "vLLM PID: $VLLM_PID"

    # Wait for the server to be ready
    if ! wait_for_server; then
        echo "Failed to start vLLM server for $MODEL_NAME, skipping..."
        continue
    fi

    # Evaluate on babilong dataset
    DATASET_NAME="RMT-team/babilong"
    TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
    LENGTHS=("64k" "128k")

    USE_CHAT_TEMPLATE=true
    USE_INSTRUCTION=true
    USE_EXAMPLES=true
    USE_POST_PROMPT=true

    echo "Running $MODEL_NAME on ${TASKS[@]} with ${LENGTHS[@]}"

    # Run the Python script
    python eval/babilong/run_model_on_babilong.py \
        --results_folder "$RESULTS_FOLDER" \
        --dataset_name "$DATASET_NAME" \
        --model_name "$MODEL_NAME" \
        --model_path "$MODEL_PATH" \
        --tasks "${TASKS[@]}" \
        --lengths "${LENGTHS[@]}" \
        --system_prompt "You are a helpful assistant." \
        $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
        $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
        $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
        $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
        --api_url "${VLLM_API_URL}/completions"

    # Evaluate on babilong-1k-samples dataset
    DATASET_NAME="RMT-team/babilong-1k-samples"
    TASKS=("qa1" "qa2" "qa3" "qa4" "qa5")
    LENGTHS=("0k" "1k" "2k" "4k" "8k" "16k" "32k")

    USE_CHAT_TEMPLATE=true
    USE_INSTRUCTION=true
    USE_EXAMPLES=true
    USE_POST_PROMPT=true

    echo "Running $MODEL_NAME on ${TASKS[@]} with ${LENGTHS[@]}"

    python eval/babilong/run_model_on_babilong.py \
        --results_folder "$RESULTS_FOLDER" \
        --dataset_name "$DATASET_NAME" \
        --model_name "$MODEL_NAME" \
        --model_path "$MODEL_PATH" \
        --tasks "${TASKS[@]}" \
        --lengths "${LENGTHS[@]}" \
        --system_prompt "You are a helpful assistant." \
        $( [ "$USE_CHAT_TEMPLATE" == true ] && echo "--use_chat_template" ) \
        $( [ "$USE_INSTRUCTION" == true ] && echo "--use_instruction" ) \
        $( [ "$USE_EXAMPLES" == true ] && echo "--use_examples" ) \
        $( [ "$USE_POST_PROMPT" == true ] && echo "--use_post_prompt" ) \
        --api_url "${VLLM_API_URL}/completions"

    # Cleanup vLLM server for this model
    cleanup

    # Wait a bit to ensure port is freed before next model
    sleep 10
done

echo "All models processed!"