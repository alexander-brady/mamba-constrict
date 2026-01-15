#!/bin/bash
#SBATCH --account=a163
#SBATCH --job-name=submit_arr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=0-3%4
# Array: 0-1 = criterions x lambdas; 2 = default (no regularization); 3 = baseline eval only

set -euo pipefail
mkdir -p logs

# If EVAL_ONLY is set to true, only run evaluation on trained models
export EVAL_ONLY=${EVAL_ONLY:-false}

MODEL_SIZE=2.8b
BASE_MODEL="state-spaces/mamba-${MODEL_SIZE}-hf"
STEPS=1k

CRITERIONS=("l2" "temporal_drift")
LAMBDAS=(0.001)

NUM_LAMBDAS=${#LAMBDAS[@]}
NUM_CRITERIONS=${#CRITERIONS[@]}
GRID_SIZE=$((NUM_CRITERIONS * NUM_LAMBDAS))
DEFAULT_ID=${GRID_SIZE}
BASELINE_ID=$((GRID_SIZE + 1))

# ---- BASELINE: eval only ----
if [ "${SLURM_ARRAY_TASK_ID}" -eq "${BASELINE_ID}" ]; then
    sbatch scripts/cscs/arr_eval.sh "${BASE_MODEL}" "true"
    exit 0
fi

# ---- Resolve criterion / lambda ----
if [ "${SLURM_ARRAY_TASK_ID}" -eq "${DEFAULT_ID}" ]; then
    CRITERION="default"
    LAMBDA=0.0
    PUSH_TO_HUB="false"
else
    CRITERION_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_LAMBDAS))
    LAMBDA_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_LAMBDAS))
    CRITERION=${CRITERIONS[$CRITERION_INDEX]}
    LAMBDA=${LAMBDAS[$LAMBDA_INDEX]}
    PUSH_TO_HUB="true"
fi

RUN_ID="mamba-${MODEL_SIZE}_${CRITERION}_w${LAMBDA}_${STEPS}"

echo "Submitting jobs for ${RUN_ID} at $(date)"

# ---- Check if only evaluation is requested ----
if [ "${EVAL_ONLY}" = "true" ]; then
    sbatch scripts/cscs/arr_eval.sh "${RUN_ID}"
    echo "Submitted EVAL only for ${RUN_ID}"
    exit 0
fi

# ---- Submit training job ----
TRAIN_JOBID=$(sbatch --parsable \
    scripts/cscs/arr_train.sh \
    "$RUN_ID" "$CRITERION" "$LAMBDA" "$MODEL_SIZE" "$PUSH_TO_HUB")

# ---- Submit evaluation job dependent on training ----
sbatch --dependency=afterok:${TRAIN_JOBID} \
    scripts/cscs/arr_eval.sh "${RUN_ID}"

echo "Submitted TRAIN=${TRAIN_JOBID} → EVAL (afterok)"