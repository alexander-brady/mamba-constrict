#!/bin/bash
#SBATCH --account=large-sc-2
#SBATCH --job-name=data_download
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --environment=finetune
#SBATCH --no-requeue # Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs.
#SBATCH -C thp_never&nvidia_vboost_enabled

echo "Beginning downloading data at $(date)"

source ./scripts/cscs/env.sh

pg19_dir="$PROJECT_DIR/data/pg19"

# Check if gsutil is installed
if ! command -v gsutil &> /dev/null; then
    pip install gsutil
fi

# Create data directory if it doesn't exist
mkdir -p "$pg19_dir"
pushd "$pg19_dir"

if [ -d "test" ] && [ -d "train" ] && [ -d "validation" ]; then
    echo "Data already exists in $pg19_dir, skipping download."
else
    # Download the datasets using gsutil
    echo "Downloading datasets from gs://deepmind-gutenberg/..."
    gsutil -m cp -r \
      "gs://deepmind-gutenberg/test" \
      "gs://deepmind-gutenberg/train" \
      "gs://deepmind-gutenberg/validation" \
      $pg19_dir/
      echo "DeepMind Gutenberg dataset downloaded to $pg19_dir."
fi

popd

python -m finetune.cli.prepare_data \
    data.data_dir=$PROJECT_DIR/data \
    hydra.run.dir="$PROJECT_DIR/outputs/data_$SLURM_JOB_ID"

echo "Finished downloading data at $(date)"