#!/bin/bash
#SBATCH --account=a163
#SBATCH --job-name=data_download
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --environment=finetune
#SBATCH --no-requeue # Prevent Slurm to requeue the job if the execution crashes (e.g. node failure) so we don't loose the logs.
#SBATCH -C thp_never&nvidia_vboost_enabled

# Usage: sbatch scripts/cscs/data.sh [DATASET]
# DATASET can be: pg19, passkey, babilong, or all (default: all)

set -euo pipefail

echo "Beginning data preparation at $(date)"

source ./scripts/cscs/env.sh

DATASET="${1:-all}"

# Function to download PG19 data
download_pg19() {
    echo "=== Downloading PG19 dataset ==="
    pg19_dir="$PROJECT_DIR/data/pg19"

    # Check if gsutil is installed
    if ! command -v gsutil &> /dev/null; then
        pip install gsutil
    fi

    mkdir -p "$pg19_dir"
    pushd "$pg19_dir" > /dev/null

    if [ -d "test" ] && [ -d "train" ] && [ -d "validation" ]; then
        echo "PG19 data already exists in $pg19_dir, skipping download."
    else
        echo "Downloading datasets from gs://deepmind-gutenberg/..."
        gsutil -m cp -r \
          "gs://deepmind-gutenberg/test" \
          "gs://deepmind-gutenberg/train" \
          "gs://deepmind-gutenberg/validation" \
          "$pg19_dir/"
        echo "DeepMind Gutenberg dataset downloaded to $pg19_dir."
    fi

    popd > /dev/null
}

# Function to prepare PG19 data
prepare_pg19() {
    echo "=== Preparing PG19 dataset ==="
    download_pg19
    echo "PG19 preparation complete."
}

# Function to prepare Passkey data
prepare_passkey() {
    echo "=== Preparing Passkey dataset ==="
    python -m finetune.cli.prepare_data \
        data=passkey \
        hydra.run.dir="$PROJECT_DIR/outputs/data_passkey_$SLURM_JOB_ID"
    echo "Passkey preparation complete."
}

# Function to prepare Babilong data
prepare_babilong() {
    echo "=== Preparing Babilong dataset ==="
    python -m finetune.cli.prepare_data \
        data=babilong \
        hydra.run.dir="$PROJECT_DIR/outputs/data_babilong_$SLURM_JOB_ID"
    echo "Babilong preparation complete."
}

# Function to prepare The Pile data
prepare_pile() {
    echo "=== Preparing The Pile dataset ==="
    python -m finetune.cli.prepare_data \
        data=the_pile \
        hydra.run.dir="$PROJECT_DIR/outputs/data_pile_$SLURM_JOB_ID"
    echo "The Pile preparation complete."
}

# Main logic
case "$DATASET" in
    pg19)
        prepare_pg19
        ;;
    passkey)
        prepare_passkey
        ;;
    babilong)
        prepare_babilong
        ;;
    pile)
        prepare_pile
        ;;
    all)
        prepare_pg19
        prepare_passkey
        prepare_babilong
        prepare_pile
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        echo "Usage: sbatch scripts/cscs/data.sh [pg19|passkey|babilong|pile|all]"
        exit 1
        ;;
esac

echo "Finished data preparation at $(date)"