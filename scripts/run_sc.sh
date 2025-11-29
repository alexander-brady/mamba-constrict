#!/bin/bash
#SBATCH --job-name=finetune_mamba
#SBATCH --account=deep_learning
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --tmp=40G
#SBATCH --mem-per-cpu=24G
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpumem:16g
#SBATCH --mail-type=END,FAIL

echo "Beginning finetuning at $(date)"

source scripts/env_sc.sh

uv run -m finetune

echo "Finished finetuning at $(date)"
