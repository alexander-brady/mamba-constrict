#!/bin/bash
#SBATCH --account=a163
#SBATCH --job-name=babilong
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --environment=eval
#SBATCH --no-requeue
#SBATCH -C thp_never&nvidia_vboost_enabled

set -euo pipefail
mkdir -p logs


MODEL_NAME="$1"

# if second arg is set to hf, it's a hf model name
# Otherwise, it's a path under models/
VERSION="${2:-"base"}"
if [ "$VERSION" = "hf" ]; then
    MODEL_PATH="${MODEL_NAME}"
elif [ "$VERSION" = "ft" ]; then
    MODEL_PATH="${PROJECT_DIR}/models/babilong/${MODEL_NAME}-babilong"
else
    MODEL_PATH="${PROJECT_DIR}/models/base/${MODEL_NAME}"
fi

echo "Starting BABILong evaluation of ${MODEL_NAME} at $(date)"
echo "MODEL_PATH=${MODEL_PATH}"

export PYTORCH_ALLOC_CONF=expandable_segments:True

RESULTS_DIR="${PROJECT_DIR}/results/${MODEL_NAME}"
WANDB_DIR="${PROJECT_DIR}/outputs/babilong/"
mkdir -p "$RESULTS_DIR"
mkdir -p "$WANDB_DIR"

python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'Device: {torch.cuda.get_device_name(0)}')"

pushd eval/babilong > /dev/null
python3 run_model_on_babilong.py \
    --results_folder "$RESULTS_DIR" \
    --dataset_name "RMT-team/babilong" \
    --model_name "$MODEL_NAME" \
    --model_path "$MODEL_PATH" \
    --tasks qa1 qa2 qa3 qa4 qa5 qa6 qa7 qa8 qa9 qa10 \
    --lengths 2k 4k 8k 16k 32k 64k 128k 256k 512k 1M \
    --use_instruction \
    --use_examples \
    --use_post_prompt \
    --wandb_project "eval-mamba" \
    --wandb_entity "mamba-monks" \
    --wandb_name "${MODEL_NAME}" \
    --wandb_dir "${WANDB_DIR}"

python3 result.py --results_dir "$RESULTS_DIR" --model_name "$MODEL_NAME"
popd > /dev/null

echo "Finished BABILong evaluation at $(date)"
