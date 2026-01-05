#!/bin/sh
#SBATCH --job-name=vllm_docker_build
#SBATCH --output=logs/vllm_docker_build_%j.out
#SBATCH --error=logs/vllm_docker_build_%j.err
#SBATCH --time=01:00:00
#SBATCH --account=large-sc-2

set -eu

mkdir -p logs

podman build \
  --build-arg PROJECT_DIR="$SCRATCH/finetune" \
  -t vllm:latest \
  -f slurm/Dockerfile.vllm \
  .

rm -f slurm/vllm.sqsh
enroot import -o slurm/vllm.sqsh podman://vllm:latest

