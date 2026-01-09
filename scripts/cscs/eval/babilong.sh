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

MODEL_PATH="$1"
MODEL_NAME="$(basename "${MODEL_PATH}")"

echo "Starting BABILong evaluation of ${MODEL_NAME} at $(date)"
echo "MODEL_PATH=${MODEL_PATH}"

export PYTORCH_ALLOC_CONF=expandable_segments:True

RESULTS_DIR="${PROJECT_DIR}/results/${MODEL_NAME}"
mkdir -p "$RESULTS_DIR"

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
    --wandb_name "${MODEL_NAME}"

python3 result.py --results_dir "$RESULTS_DIR" --model_name "$MODEL_NAME"
popd > /dev/null

echo "Finished BABILong evaluation at $(date)"
