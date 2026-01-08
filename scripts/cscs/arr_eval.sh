#!/bin/bash
#SBATCH --account=a163
#SBATCH --job-name=eval
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

echo "Starting LM-eval of ${MODEL_NAME} at $(date)"
echo "MODEL_PATH=${MODEL_PATH}"

MODEL_ARGS="pretrained=${MODEL_PATH} tokenizer=state-spaces/mamba-2.8b-hf tensor_parallel_size=4 dtype=bfloat16 trust_remote_code=True max_model_len=32768"
WANDB_ARGS="project=eval-mamba entity=mamba-monks name=${MODEL_NAME} dir=${PROJECT_DIR}/outputs/lm-eval/"

lm-eval run \
    --config "${PROJECT_DIR}/eval/lm-eval.yaml" \
    --model_args "${MODEL_ARGS}" \
    --output_path "${PROJECT_DIR}/results/lm-eval/${MODEL_NAME}" \
    --wandb_args "${WANDB_ARGS}" \
    --tasks lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa

echo "Finished LM-eval evaluation at $(date)"
