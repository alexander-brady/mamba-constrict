#!/bin/bash
#SBATCH --account=a-a06
#SBATCH --job-name=finetune_4_mamba
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --tmp=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --environment=slurm/finetune.toml
#SBATCH --no-requeue # Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs.
#SBATCH -C thp_never&nvidia_vboost_enabled

echo "Beginning finetuning at $(date)"

export TOKENIZERS_PARALLELISM=false  # Disable tokenizer parallelism to avoid deadlocks

uv run -m finetune

echo "Finished finetuning at $(date)"
