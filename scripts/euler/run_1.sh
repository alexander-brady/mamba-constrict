#!/bin/bash
#SBATCH --job-name=finetune_1_mamba
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --tmp=64G
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gpus=a100_80gb:1
#SBATCH --mail-type=END,FAIL

echo "Beginning finetuning at $(date)"

export TOKENIZERS_PARALLELISM=false  # Disable tokenizer parallelism to avoid deadlocks

source scripts/euler/env.sh

uv run -m finetune model.name=state-spaces/mamba-130m-hf

echo "Finished finetuning at $(date)"
