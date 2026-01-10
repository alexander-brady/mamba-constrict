#!/bin/bash
#SBATCH --account=a163
#SBATCH --job-name=pg19_filter
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=320G
#SBATCH --environment=finetune_old
#SBATCH --no-requeue
#SBATCH -C thp_never&nvidia_vboost_enabled
# Filter PG19 books that have at least 128k tokens
# Uses the same tokenizer setup as run_perplexity.py

set -euo pipefail

# Default values
MODEL_SIZE=2.8b
MODEL="state-spaces/mamba-${MODEL_SIZE}-hf"
DATA_DIR="${1:-./data/pg19/test}"
OUTPUT_DIR="${2:-${DATA_DIR}_128k}"
MIN_TOKENS="${3:-131072}"  # 128k = 131072 tokens

echo "========================================"
echo "Filter PG19 books >= ${MIN_TOKENS} tokens"
echo "========================================"
echo "Model/Tokenizer: ${MODEL}"
echo "Input directory: ${DATA_DIR}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Minimum tokens: ${MIN_TOKENS}"
echo ""

# Check input directory
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' does not exist."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run Python script to filter books
python3 - "$MODEL" "$DATA_DIR" "$OUTPUT_DIR" "$MIN_TOKENS" << 'PYTHON_SCRIPT'
import sys
import os
from pathlib import Path
from tqdm import tqdm
import shutil

# Parse arguments
model_name = sys.argv[1]
data_dir = Path(sys.argv[2])
output_dir = Path(sys.argv[3])
min_tokens = int(sys.argv[4])

# Import tokenizer (same setup as run_perplexity.py)
from transformers import AutoTokenizer

print(f"Loading tokenizer from {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Get all text files
text_files = sorted(list(data_dir.glob("*.txt")))
print(f"Found {len(text_files)} text files")

# Filter books
count = 0
for txt_file in tqdm(text_files, desc="Filtering books"):
    with open(txt_file, encoding="utf-8") as f:
        text = f.read()

    if not text.strip():
        continue

    # Tokenize (same as run_perplexity.py - no special tokens, then add BOS if available)
    input_ids = tokenizer.encode(text, add_special_tokens=False)
    num_tokens = len(input_ids)
    if tokenizer.bos_token_id is not None:
        num_tokens += 1

    if num_tokens >= min_tokens:
        # Copy file to output directory
        shutil.copy2(txt_file, output_dir / txt_file.name)
        count += 1

print(f"\n{'=' * 40}")
print(f"Books with >= {min_tokens} tokens: {count}")
print(f"Output directory: {output_dir}")
print(f"{'=' * 40}")
PYTHON_SCRIPT

# Count and report
NUM_BOOKS=$(find "$OUTPUT_DIR" -name "*.txt" | wc -l | tr -d ' ')
echo ""
echo "========================================"
echo "RESULT: ${NUM_BOOKS} books with >= ${MIN_TOKENS} tokens"
echo "Saved to: ${OUTPUT_DIR}"
echo "========================================"
