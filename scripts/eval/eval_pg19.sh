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
#SBATCH --gres=gpumem:64g
#SBATCH --mail-type=END,FAIL

# Usage: sbatch scripts/eval/eval_pg19.sh [data_dir]
# Or: ./scripts/eval/eval_pg19.sh [data_dir]

# Get PG19 test data directory from argument or environment variable
DATA_DIR=${1:-${SCRATCH}/finetune/data/pg19/test}

# Source environment
if [ -f "scripts/env.sh" ]; then
    source scripts/env.sh
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

# Get list of models from config
MODEL_NAMES=$(python3 -c 'import json; print(" ".join([k for k in json.load(open("eval/config/model2path.json")).keys() if not k.startswith("_")]))')

# Context lengths to evaluate (powers of 2)
CONTEXT_LENGTHS="512 1024 2048 4096 8192 16384 32768 65536 131072"

for MODEL_NAME in $MODEL_NAMES; do
    MODEL_PATH=$(python3 -c "import json; print(json.load(open('eval/config/model2path.json'))['$MODEL_NAME'])")
    MAX_LEN=$(python3 -c "import json; print(json.load(open('eval/config/model2maxlen.json'))['$MODEL_NAME'])")

    echo "================================================================"
    echo "Processing Model: $MODEL_NAME"
    echo "Model Path: $MODEL_PATH"
    echo "Max Length: $MAX_LEN"
    echo "================================================================"

    # Filter context lengths based on max model length
    VALID_CONTEXT_LENGTHS=""
    for CTX in $CONTEXT_LENGTHS; do
        if [ "$CTX" -le "$MAX_LEN" ]; then
            VALID_CONTEXT_LENGTHS="$VALID_CONTEXT_LENGTHS $CTX"
        fi
    done

    echo "Valid context lengths for this model: $VALID_CONTEXT_LENGTHS"

    # Run perplexity evaluation
    pushd eval/pg19 > /dev/null
    python run_perplexity.py \
        --model "$MODEL_NAME" \
        --data_dir "$DATA_DIR" \
        --save_dir "../../$RESULTS_FOLDER" \
        --context_lengths $VALID_CONTEXT_LENGTHS \
        --stride 512 \
        --batch_size 1

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
