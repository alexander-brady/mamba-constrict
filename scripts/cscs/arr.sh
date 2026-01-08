#!/bin/bash
#SBATCH --account=a163
#SBATCH --job-name=train_eval
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=0-25%4 
# Array: 0-23: combinations of 4 criterions x 6 lambdas; 24: default (no regularization); 25: baseline eval only

set -euo pipefail
mkdir -p logs

MODEL_SIZE=2.8b
BASE_MODEL="state-spaces/mamba-${MODEL_SIZE}-hf"

# ---- BASELINE: eval only ----
if [ ${SLURM_ARRAY_TASK_ID} -eq 25 ]; then
    sbatch scripts/cscs/arr_eval.sh "${BASE_MODEL}"
    exit 0
fi

CRITERIONS=("l1" "l2" "mahalanobis" "temporal_drift")
LAMBDAS=(0.01 0.1 0.3 0.5 1.0 2.0)

# ---- Resolve criterion / lambda ----
if [ ${SLURM_ARRAY_TASK_ID} -eq 24 ]; then
    CRITERION="default"
    LAMBDA=0.0
else
    CRITERION_INDEX=$((SLURM_ARRAY_TASK_ID / 6))
    LAMBDA_INDEX=$((SLURM_ARRAY_TASK_ID % 6))
    CRITERION=${CRITERIONS[$CRITERION_INDEX]}
    LAMBDA=${LAMBDAS[$LAMBDA_INDEX]}
fi

RUN_ID="mamba-${MODEL_SIZE}_${CRITERION}_w${LAMBDA}"

echo "Submitting jobs for ${RUN_ID} at $(date)"

# ---- Submit training job ----
TRAIN_JOBID=$(sbatch --parsable \
    scripts/cscs/arr_train.sh \
    "$RUN_ID" "$CRITERION" "$LAMBDA" "$MODEL_SIZE")

# ---- Submit evaluation job dependent on training ----
sbatch \
    --dependency=afterok:${TRAIN_JOBID} \
    scripts/cscs/arr_eval.sh "models/${RUN_ID}"

echo "Submitted TRAIN=${TRAIN_JOBID} → EVAL (afterok)"
