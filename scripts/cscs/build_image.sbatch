#!/bin/sh
#SBATCH --job-name=docker_build
#SBATCH --output=logs/docker_build.out
#SBATCH --error=logs/docker_build.err
#SBATCH --time=01:00:00
#SBATCH --account=large-sc-2

set -eu

mkdir -p logs

podman build -t finetune:0.1-py3 -f slurm/Dockerfile

rm -f slurm/finetune.sqsh
enroot import -o slurm/finetune.sqsh podman://finetune:0.1-py3