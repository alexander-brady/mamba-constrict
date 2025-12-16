#!/bin/bash
#SBATCH --job-name=pg19_eval
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

# Usage: sbatch scripts/eval/eval_pg19.sh [raw_data_dir]
# Or: ./scripts/eval/eval_pg19.sh [raw_data_dir]
# Note: Expects raw .txt files from PG19 test split (not preprocessed data)

# Get PG19 raw test data directory from argument or environment variable
# This should point to the folder containing raw .txt files
DATA_DIR=${1:-${SCRATCH}/finetune/data/pg19/test}

# Source environment
if [ -f "scripts/euler/env.sh" ]; then
    source scripts/euler/env.sh
fi

echo "PG19 Perplexity Evaluation"
echo "Data directory: $DATA_DIR"

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' does not exist."
    echo "Please run data preparation first or provide the correct path."
    exit 1
fi

# Results folder
RESULTS_FOLDER="./results/pg19"
mkdir -p "$RESULTS_FOLDER"

# Get list of models using model_utils
MODEL_NAMES=$(python3 -c 'import sys; sys.path.insert(0, "eval"); from model_utils import get_all_models; print(" ".join(get_all_models().keys()))')

# Context lengths to evaluate (powers of 2, up to 256k)
CONTEXT_LENGTHS="2048 4096 8192 16384 32768 65536 131072 262144"

for MODEL_NAME in $MODEL_NAMES; do
    MODEL_PATH=$(python3 -c "import sys; sys.path.insert(0, 'eval'); from model_utils import get_all_models; print(get_all_models()['$MODEL_NAME'])")

    echo "================================================================"
    echo "Processing Model: $MODEL_NAME"
    echo "Model Path: $MODEL_PATH"
    echo "================================================================"

    # For SSMs, no need to filter context lengths (no hard limit)
    VALID_CONTEXT_LENGTHS="$CONTEXT_LENGTHS"
    echo "Context lengths: $VALID_CONTEXT_LENGTHS"

    # Run perplexity evaluation
    pushd eval/pg19 > /dev/null
    python run_perplexity.py \
        --model "$MODEL_PATH" \
        --data_dir "$DATA_DIR" \
        --save_dir "../../$RESULTS_FOLDER" \
        --context_lengths $VALID_CONTEXT_LENGTHS \
        --stride 512

    popd > /dev/null

    echo ""
done

# Generate plots
echo "================================================================"
echo "Generating plots..."
echo "================================================================"

pushd eval/pg19 > /dev/null
python plot_results.py \
    --results_dir "../../$RESULTS_FOLDER" \
    --title "PG19"
popd > /dev/null

echo ""
echo "================================================================"
echo "All models processed!"
echo "Results saved to: $RESULTS_FOLDER"
echo "================================================================"
