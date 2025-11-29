#!/bin/bash
#SBATCH --job-name=data_download
#SBATCH --account=deep_learning
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --tmp=40G
#SBATCH --mem-per-cpu=24G
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL

echo "Beginning downloading data at $(date)"

source scripts/env_sc.sh

uv run -m finetune.cli.prepare_data

echo "Finished downloading data at $(date)"
