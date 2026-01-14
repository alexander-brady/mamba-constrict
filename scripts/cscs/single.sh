#!/bin/bash
#SBATCH --account=a163
#SBATCH --job-name=submit_single
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail
mkdir -p logs

# -----------------------
# Usage:
#   sbatch submit_single.sh [CRITERION] [LAMBDA] [MODEL_SIZE] [STEPS]
#
# Examples:
#   sbatch submit_single.sh temporal_drift 0.001 2.8b 2k
#   sbatch --export=EVAL_ONLY=true submit_single.sh temporal_drift 0.001 2.8b 2k
#   sbatch submit_single.sh default 0.0 2.8b 2k
# -----------------------

# If EVAL_ONLY is set to true, only run evaluation on trained models
export EVAL_ONLY="${EVAL_ONLY:-false}"

CRITERION="${1:-temporal_drift}"
LAMBDA="${2:-0.0003}"
MODEL_SIZE="${3:-2.8b}"
STEPS="${4:-2k}"

RUN_ID="mamba-${MODEL_SIZE}_${CRITERION}_w${LAMBDA}_${STEPS}"

echo "Submitting jobs for ${RUN_ID} at $(date)"
echo "  CRITERION=${CRITERION}"
echo "  LAMBDA=${LAMBDA}"
echo "  MODEL_SIZE=${MODEL_SIZE}"
echo "  STEPS=${STEPS}"
echo "  EVAL_ONLY=${EVAL_ONLY}"

# ---- Eval only ----
if [ "${EVAL_ONLY}" = "true" ]; then
  sbatch scripts/cscs/arr_eval.sh "${RUN_ID}"
  echo "Submitted EVAL only for ${RUN_ID}"
  exit 0
fi

# ---- Submit training job ----
TRAIN_JOBID=$(sbatch --parsable \
  scripts/cscs/arr_train.sh \
  "${RUN_ID}" "${CRITERION}" "${LAMBDA}" "${MODEL_SIZE}")

# ---- Submit evaluation job dependent on training ----
sbatch --dependency=afterok:${TRAIN_JOBID} \
  scripts/cscs/arr_eval.sh "${RUN_ID}"

echo "Submitted TRAIN=${TRAIN_JOBID} → EVAL (afterok)"