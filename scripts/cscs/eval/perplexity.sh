#!/bin/bash
#SBATCH --account=a163
#SBATCH --time=12:00:00
#SBATCH --job-name=perplexity-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --environment=finetune_old
#SBATCH --no-requeue # Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs.
#SBATCH -C thp_never&nvidia_vboost_enabled

echo "Starting Perplexity evaluation at $(date)"

export PYTORCH_ALLOC_CONF=expandable_segments:True

cd /iopsstor/scratch/cscs/teilers/finetune
PROJECT_DIR=$(pwd)

# pip install transformers accelerate

# Create results directory
RESULTS_DIR="$PROJECT_DIR/results/perplexity"
mkdir -p "$RESULTS_DIR"

# Print pytorch version and device
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'Device: {torch.cuda.get_device_name(0)}')"

# Get list of models using model_utils
MODEL_NAMES=$(python3 -c 'import sys; sys.path.insert(0, "eval"); from model_utils import get_all_models; print(" ".join(get_all_models().keys()))')

for MODEL_NAME in $MODEL_NAMES; do
    MODEL_PATH=$(python3 -c "import sys; sys.path.insert(0, 'eval'); from model_utils import get_all_models; print(get_all_models()['$MODEL_NAME'])")

    echo "----------------------------------------------------------------"
    echo "Processing Model: $MODEL_NAME"
    echo "Model Path: $MODEL_PATH"
    echo "----------------------------------------------------------------"

    # Run inference with local model
    pushd eval/pg19 > /dev/null
    python3 run_perplexity.py --model "$MODEL_PATH" --data_dir "$PROJECT_DIR/data/pg19/test" --save_dir "$RESULTS_DIR" --context_lengths 2048 4096 8192 16384 32768 65536 131072 262144
    
    # Plot results
    python3 plot_results.py --results_dir "$RESULTS_DIR"
    popd > /dev/null
done

echo "Finished Passkey evaluation at $(date)"
