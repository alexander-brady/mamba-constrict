#!/bin/bash
#SBATCH --job-name=passkey_eval
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

# Usage: sbatch scripts/eval/eval_passkey.sh
# Or: ./scripts/eval/eval_passkey.sh

# Source environment
if [ -f "scripts/euler/env.sh" ]; then
    source scripts/euler/env.sh
fi

# Results folder
RESULTS_FOLDER="./results/passkey_retrieval"

# Get list of models using model_utils
MODEL_NAMES=$(python3 -c 'import sys; sys.path.insert(0, "eval"); from model_utils import get_all_models; print(" ".join(get_all_models().keys()))')

for MODEL_NAME in $MODEL_NAMES; do
    MODEL_PATH=$(python3 -c "import sys; sys.path.insert(0, 'eval'); from model_utils import get_all_models; print(get_all_models()['$MODEL_NAME'])")

    echo "================================================================"
    echo "Processing Model: $MODEL_NAME"
    echo "Model Path: $MODEL_PATH"
    echo "================================================================"

    # Run passkey retrieval test
    pushd eval/passkey_retrieval > /dev/null
    python run_test.py \
        --model "$MODEL_PATH" \
        --save_dir ../../$RESULTS_FOLDER \
        --token_lengths 2048 4096 8192 16384 32768 65536 131072 262144 \
        --num_tests 50

    # Export results
    python result.py --results_dir ../../$RESULTS_FOLDER
    popd > /dev/null
done

echo "All models processed!"
echo "Results saved to: $RESULTS_FOLDER"
