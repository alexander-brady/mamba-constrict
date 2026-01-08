#!/bin/bash
#SBATCH --account=a163
#SBATCH --time=12:00:00
#SBATCH --job-name=babilong-eval
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

echo "Starting BABILong evaluation at $(date)"

export PYTORCH_ALLOC_CONF=expandable_segments:True

# Create results directory
RESULTS_DIR="$PROJECT_DIR/results/babilong"
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
    pushd eval/babilong > /dev/null
    python3 run_model_on_babilong.py \
        --results_folder "$RESULTS_DIR" \
        --dataset_name "RMT-team/babilong" \
        --model_name "$MODEL_NAME" \
        --model_path "$MODEL_PATH" \
        --tasks qa1 qa2 qa3 qa4 qa5 qa6 qa7 qa8 qa9 qa10\
        --lengths 2k 4k 8k 16k 32k 64k 128k 256k 512k 1M\
        --use_instruction \
        --use_examples
    popd > /dev/null
done

echo "Finished BABILong evaluation at $(date)"
