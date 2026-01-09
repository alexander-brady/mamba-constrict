#!/bin/bash
#SBATCH --account=a163
#SBATCH --job-name=passkey
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

echo "Starting Passkey evaluation of ${MODEL_NAME} at $(date)"
echo "MODEL_PATH=${MODEL_PATH}"

export PYTORCH_ALLOC_CONF=expandable_segments:True

RESULTS_DIR="${PROJECT_DIR}/results/${MODEL_NAME}"
mkdir -p "$RESULTS_DIR"


pushd eval/passkey_retrieval > /dev/null
python3 run_test.py \
    --model "$MODEL_PATH" \
    --save_dir "$RESULTS_DIR" \
    --token_lengths 2048 4096 8192 16384 32768 65536 131072 262144 \
    --num_tests 50 \
    --wandb_project "eval-mamba" \
    --wandb_entity "mamba-monks" \
    --wandb_name "${MODEL_NAME}"

python3 result.py --results_dir "$RESULTS_DIR" --model_name "$MODEL_NAME"
popd > /dev/null

echo "Finished Passkey evaluation at $(date)"
