#!/bin/bash
#SBATCH --job-name=babilong_eval
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

# Usage: sbatch scripts/eval/babilong.sh
# Or: ./scripts/eval/babilong.sh

# Source environment
if [ -f "scripts/euler/env.sh" ]; then
    source scripts/euler/env.sh
fi

RESULTS_FOLDER="./results/babilong"
mkdir -p "$RESULTS_FOLDER"

# Get list of models from config
MODEL_NAMES=$(python3 -c 'import json; print(" ".join([k for k in json.load(open("eval/config/model2path.json")).keys() if not k.startswith("_")]))')

for MODEL_NAME in $MODEL_NAMES; do
    MODEL_PATH=$(python3 -c "import json; print(json.load(open('eval/config/model2path.json'))['$MODEL_NAME'])")

    echo "================================================================"
    echo "Processing Model: $MODEL_NAME"
    echo "Model Path: $MODEL_PATH"
    echo "================================================================"

    # Evaluate on babilong dataset
    DATASET_NAME="RMT-team/babilong"
    TASKS=("qa1" "qa2" "qa3" "qa4" "qa5" "qa6" "qa7" "qa8" "qa9" "qa10")
    LENGTHS=("2k" "4k" "8k" "16k" "32k" "64k" "128k" "256k" "512k" "1M")

    USE_CHAT_TEMPLATE=true
    USE_INSTRUCTION=true
    USE_EXAMPLES=true
    USE_POST_PROMPT=true

    echo "Running $MODEL_NAME on ${TASKS[@]} with ${LENGTHS[@]}"

    # Run the Python script (using local model, not API)
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
        --api_url ""

    echo ""
done

echo "================================================================"
echo "All models processed!"
echo "Results saved to: $RESULTS_FOLDER"
echo "================================================================"