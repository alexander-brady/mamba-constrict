#!/bin/bash
#SBATCH --job-name=data_download
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --tmp=64G
#SBATCH --mem-per-cpu=64G
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL

echo "Beginning downloading data at $(date)"

source scripts/euler/env.sh

dir="$SCRATCH/finetune/data/pg19"

# Check if gsutil is installed
    if ! command -v gsutil &> /dev/null; then
        uv pip install gsutil
    fi

# Create data directory if it doesn't exist
mkdir -p "$dir"
pushd "$dir"

if [ -d "test" ] && [ -d "train" ] && [ -d "validation" ]; then
    echo "Data already exists in $dir, skipping download."
else
    # Download the datasets using gsutil
    echo "Downloading datasets from gs://deepmind-gutenberg/..."
    gsutil -m cp -r \
      "gs://deepmind-gutenberg/test" \
      "gs://deepmind-gutenberg/train" \
      "gs://deepmind-gutenberg/validation" \
      .
fi

popd

uv run -m finetune.cli.prepare_data

echo "Finished downloading data at $(date)"
