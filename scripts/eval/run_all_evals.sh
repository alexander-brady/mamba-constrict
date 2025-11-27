#!/usr/bin/env bash
# Master script to run all evaluation benchmarks
# Usage: ./scripts/eval/run_all_evals.sh [gpu_memory_utilization] [tensor_parallel_size]
# Example: CUDA_VISIBLE_DEVICES=0,1 TP=2 ./scripts/eval/run_all_evals.sh 0.95 2

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
if bash scripts/eval/eval_longbench.sh "$GPU_MEMORY_UTILIZATION" "$TENSOR_PARALLEL_SIZE"; then
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
if bash scripts/eval/eval_babilong.sh "$TENSOR_PARALLEL_SIZE"; then
    echo "✓ Babilong evaluation completed successfully"
else
    echo "✗ Babilong evaluation failed"
    BABILONG_FAILED=1
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

if [ -z "$LONGBENCH_FAILED" ] && [ -z "$BABILONG_FAILED" ]; then
    echo "✓ All evaluations completed successfully!"
    echo ""
    echo "Results location:"
    echo "  - LongBench: results/longbench/"
    echo "  - Babilong:  results/babilong/"
    exit 0
else
    echo "Some evaluations failed:"
    [ -n "$LONGBENCH_FAILED" ] && echo "  ✗ LongBench"
    [ -n "$BABILONG_FAILED" ] && echo "  ✗ Babilong"
    exit 1
fi
