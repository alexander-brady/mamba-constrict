#!/bin/bash
#SBATCH --account=a163
#SBATCH --job-name=perplexity
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --environment=finetune_old
#SBATCH --no-requeue
#SBATCH -C thp_never&nvidia_vboost_enabled

set -euo pipefail
mkdir -p logs

PROJECT_DIR="/users/teilers/scratch/finetune"
MODEL_NAME="$1"

# if second arg is set to hf, it's a hf model name
# Otherwise, it's a path under models/
VERSION="${2:-"base"}"
if [ "$VERSION" = "hf" ]; then
    MODEL_PATH="${MODEL_NAME}"
else
    MODEL_PATH="models/base/${MODEL_NAME}"
fi

cd "$PROJECT_DIR"

echo "Starting Perplexity evaluation of ${MODEL_NAME} at $(date)"
echo "MODEL_PATH=${MODEL_PATH}"

export PYTORCH_ALLOC_CONF=expandable_segments:True

RESULTS_DIR="${PROJECT_DIR}/results/${MODEL_NAME}"
WANDB_DIR="${PROJECT_DIR}/outputs/perplexity/"
mkdir -p "$RESULTS_DIR"
mkdir -p "$WANDB_DIR"

python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'Device: {torch.cuda.get_device_name(0)}')"

pushd eval/pg19 > /dev/null
python3 run_perplexity.py \
    --model "$MODEL_PATH" \
    --data_dir "${PROJECT_DIR}/data/pg19/test_128k" \
    --save_dir "$RESULTS_DIR" \
    --context_lengths 2048 4096 8192 16384 32768 65536 131072 \
    --wandb_project "eval-mamba" \
    --wandb_entity "mamba-monks" \
    --wandb_name "${MODEL_NAME}" \
    --wandb_dir "${WANDB_DIR}"

python3 plot_results.py --results_dir "$RESULTS_DIR" --model_name "$MODEL_NAME"
popd > /dev/null

echo "Finished Perplexity evaluation at $(date)"