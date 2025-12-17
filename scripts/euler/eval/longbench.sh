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

# Source environment
if [ -f "scripts/euler/env.sh" ]; then
    source scripts/euler/env.sh
fi

echo "Starting LongBench evaluation at $(date)"

# Get list of models using model_utils
MODEL_NAMES=$(python3 -c 'import sys; sys.path.insert(0, "eval"); from model_utils import get_all_models; print(" ".join(get_all_models().keys()))')

for MODEL_NAME in $MODEL_NAMES; do
    MODEL_PATH=$(python3 -c "import sys; sys.path.insert(0, 'eval'); from model_utils import get_all_models; print(get_all_models()['$MODEL_NAME'])")

    echo "----------------------------------------------------------------"
    echo "Processing Model: $MODEL_NAME"
    echo "Model Path: $MODEL_PATH"
    echo "----------------------------------------------------------------"

    # Run inference with local model
    pushd eval/LongBench > /dev/null
    python pred.py --model "$MODEL_PATH" --save_dir ../../results/longbench

    # Export results
    python result.py --results_dir ../../results/longbench
    popd > /dev/null
done

echo "Finished LongBench evaluation at $(date)"
