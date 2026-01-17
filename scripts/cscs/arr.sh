#!/bin/bash
#SBATCH --account=a163
#SBATCH --job-name=submit_arr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --array=0-5%4 # 4x lambdas + default (no baseline)

set -euo pipefail
mkdir -p logs

# If EVAL_ONLY is set to true, only run evaluation on trained models
export EVAL_ONLY=${EVAL_ONLY:-false}

BASELINE_ID=-1 # No baseline
DEFAULT_ID=5

MODEL_SIZE=2.8b
BASE_MODEL="state-spaces/mamba-${MODEL_SIZE}-hf"
STEPS=300

LAMBDAS=(0.0001 0.001 0.01 0.1)

# ---- BASELINE: eval only ----
if [ "${SLURM_ARRAY_TASK_ID}" -eq "${BASELINE_ID}" ]; then
    sbatch scripts/cscs/arr_eval.sh "${BASE_MODEL}" "true"
    exit 0
fi

# ---- Resolve criterion / lambda ----
if [ "${SLURM_ARRAY_TASK_ID}" -eq "${DEFAULT_ID}" ]; then
    CRITERION="default"
    LAMBDA=0.0
    # PUSH_TO_HUB="false"
else
    CRITERION="l2"
    LAMBDA=${LAMBDAS[${SLURM_ARRAY_TASK_ID}]}
    # PUSH_TO_HUB="false"
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
    "$RUN_ID" "$CRITERION" "$LAMBDA" "$MODEL_SIZE")

# ---- Submit evaluation job dependent on training ----
sbatch --dependency=afterok:${TRAIN_JOBID} \
    scripts/cscs/arr_eval.sh "${RUN_ID}"

echo "Submitted TRAIN=${TRAIN_JOBID} → EVAL (afterok)"