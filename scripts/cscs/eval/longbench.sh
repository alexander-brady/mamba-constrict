#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH --time=12:00:00
#SBATCH --job-name=longbench-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --environment=eval
#SBATCH --no-requeue # Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs.
#SBATCH -C thp_never&nvidia_vboost_enabled

echo "Starting LongBench evaluation at $(date)"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Create results directory
RESULTS_DIR="$PROJECT_DIR/results/longbench"
mkdir -p "$RESULTS_DIR"

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
    python3 pred.py --model "$MODEL_PATH" --save_dir "$RESULTS_DIR"

    # Export results
    python3 result.py --results_dir "$RESULTS_DIR"
    popd > /dev/null
done

echo "Finished LongBench evaluation at $(date)"
