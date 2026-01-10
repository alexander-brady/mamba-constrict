#!/bin/bash
#SBATCH --account=a163
#SBATCH --job-name=finetune
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --environment=finetune
#SBATCH -C thp_never&nvidia_vboost_enabled

set -euo pipefail

MODEL_NAME="$1"

DATASET="$2"

# if third arg is set to hf, it's a hf model name
# Otherwise, it's a path under models/
VERSION="${3:-"base"}"
if [ "$VERSION" = "hf" ]; then
    MODEL_PATH="${MODEL_NAME}"
else
    MODEL_PATH="models/base/${MODEL_NAME}"
fi

RUN_ID="${MODEL_NAME}-${DATASET}"

source ./scripts/cscs/env.sh

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Beginning finetuning of ${RUN_ID} at $(date)"
echo "MODEL_PATH=${MODEL_PATH} DATASET=${DATASET}"

srun $PROJECT_DIR/.venv/bin/python -m finetune \
    run_id="${RUN_ID}" \
    data="${DATASET}" \
    trainer="evals" \
    model.name="${MODEL_PATH}" \
    wandb.project="post-finetune-mamba" \
    +wandb.job_type="${DATASET}"

echo "Finished finetuning of ${RUN_ID} at $(date)"
