#!/bin/bash
#SBATCH --job-name=eval_all
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --tmp=64G
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpumem:32G
#SBATCH --mail-type=END,FAIL

set -e

GPU_MEMORY_UTILIZATION=${1:-0.95}
TENSOR_PARALLEL_SIZE=${2:-1}

echo "========================================================================"
echo "🚀 Starting all evaluations 🚀"
echo "Remember to set the model paths in eval/config/model2path.json and"
echo "the maximum sequence lengths in eval/config/model2maxlen.json."
echo "========================================================================"
echo ""

# Create results directory
mkdir -p results

# Track start time
START_TIME=$(date +%s)

# Run LongBench evaluation
echo "========================================================================"
echo "Starting LongBench Evaluation"
echo "========================================================================"
if bash scripts/eval/longbench.sh "$GPU_MEMORY_UTILIZATION" "$TENSOR_PARALLEL_SIZE"; then
    echo "✓ LongBench evaluation completed successfully"
else
    echo "✗ LongBench evaluation failed"
    LONGBENCH_FAILED=1
fi
echo ""

# Run Babilong evaluation
echo "========================================================================"
echo "Starting Babilong Evaluation"
echo "========================================================================"
if bash scripts/eval/babilong.sh "$TENSOR_PARALLEL_SIZE"; then
    echo "✓ Babilong evaluation completed successfully"
else
    echo "✗ Babilong evaluation failed"
    BABILONG_FAILED=1
fi
echo ""

# Run Passkey Retrieval evaluation
echo "========================================================================"
echo "Starting Passkey Retrieval Evaluation"
echo "========================================================================"
if bash scripts/eval/eval_passkey.sh "$GPU_MEMORY_UTILIZATION" "$TENSOR_PARALLEL_SIZE"; then
    echo "✓ Passkey Retrieval evaluation completed successfully"
else
    echo "✗ Passkey Retrieval evaluation failed"
    PASSKEY_FAILED=1
fi
echo ""

# Run PG19 Perplexity evaluation
echo "========================================================================"
echo "Starting PG19 Perplexity Evaluation"
echo "========================================================================"
if bash scripts/eval/eval_pg19.sh; then
    echo "✓ PG19 Perplexity evaluation completed successfully"
else
    echo "✗ PG19 Perplexity evaluation failed"
    PG19_FAILED=1
fi
echo ""

# Calculate total time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

# Print summary
echo "========================================================================"
echo "Evaluation Summary"
echo "========================================================================"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""

if [ -z "$LONGBENCH_FAILED" ] && [ -z "$BABILONG_FAILED" ] && [ -z "$PASSKEY_FAILED" ] && [ -z "$PG19_FAILED" ]; then
    echo "✓ All evaluations completed successfully!"
    echo ""
    echo "Results location:"
    echo "  - LongBench:         results/longbench/"
    echo "  - Babilong:          results/babilong/"
    echo "  - Passkey Retrieval: results/passkey_retrieval/"
    echo "  - PG19 Perplexity:   results/pg19/"
    exit 0
else
    echo "Some evaluations failed:"
    [ -n "$LONGBENCH_FAILED" ] && echo "  ✗ LongBench"
    [ -n "$BABILONG_FAILED" ] && echo "  ✗ Babilong"
    [ -n "$PASSKEY_FAILED" ] && echo "  ✗ Passkey Retrieval"
    [ -n "$PG19_FAILED" ] && echo "  ✗ PG19 Perplexity"
    exit 1
fi
