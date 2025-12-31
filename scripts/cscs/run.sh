#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH --job-name=finetune
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --environment=finetune
#SBATCH --no-requeue # Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs.
#SBATCH -C thp_never&nvidia_vboost_enabled

echo "Beginning finetuning at $(date)"

source ./scripts/cscs/env.sh

export CRITERION=l1
export LAMBDA=0.1
export MODEL_SIZE=2.8b

export TOKENIZERS_PARALLELISM=false  # Disable tokenizer parallelism to avoid deadlocks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m torch.distributed.run --nproc_per_node=4 -m finetune \
    loss=$CRITERION \
    loss.weight=$LAMBDA \
    model.name=state-spaces/mamba-$MODEL_SIZE-hf \

echo "Finished finetuning at $(date)"
