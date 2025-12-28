#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH --job-name=finetune
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --environment=finetune
#SBATCH --no-requeue # Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs.
#SBATCH -C thp_never&nvidia_vboost_enabled

echo "Beginning finetuning at $(date)"

dir="$SCRATCH/finetune"
export HF_HOME="$dir/.hf/" 

export TOKENIZERS_PARALLELISM=false  # Disable tokenizer parallelism to avoid deadlocks

python -m finetune hydra.run.dir="$dir/outputs/finetune_$SLURM_JOB_ID"

echo "Finished finetuning at $(date)"
