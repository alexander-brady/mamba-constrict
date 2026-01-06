#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH --time=12:00:00
#SBATCH --job-name=lm-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --environment=eval
#SBATCH --no-requeue # Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs.
#SBATCH -C thp_never&nvidia_vboost_enabled


MODEL_PATH="state-spaces/mamba-2.8b-hf"
MODEL_NAME="${MODEL_PATH##*/}"

echo "Starting LM-eval of $MODEL_NAME at $(date)"

MODEL_ARGS="pretrained=${MODEL_PATH} tokenizer=state-spaces/mamba-2.8b-hf tensor_parallel_size=4 dtype=bfloat16 trust_remote_code=True max_model_len=32768 "
lm-eval run --config ${PROJECT_DIR}/eval/lm-eval.yaml --model_args ${MODEL_ARGS} --output_path ${PROJECT_DIR}/results/lm-eval/${MODEL_NAME}

echo "Finished LM-eval evaluation at $(date)"