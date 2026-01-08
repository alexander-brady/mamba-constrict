#!/bin/bash
#SBATCH --account=a163
#SBATCH --job-name=finetune
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --environment=finetune
#SBATCH -C thp_never&nvidia_vboost_enabled

set -euo pipefail

RUN_ID="$1"
CRITERION="$2"
LAMBDA="$3"
MODEL_SIZE="$4"

export CRITERION LAMBDA MODEL_SIZE

source ./scripts/cscs/env.sh

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Beginning finetuning of ${RUN_ID} at $(date)"
echo "CRITERION=${CRITERION} LAMBDA=${LAMBDA} MODEL_SIZE=${MODEL_SIZE}"

srun $PROJECT_DIR/.venv/bin/python -m finetune \
    run_id="${RUN_ID}" \
    loss="${CRITERION}" \
    loss.weight="${LAMBDA}" \
    model.name="state-spaces/mamba-${MODEL_SIZE}-hf"

echo "Finished finetuning of ${RUN_ID} at $(date)"
