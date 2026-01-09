#!/bin/bash
#SBATCH --account=a163
#SBATCH --job-name=eval
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G

set -euo pipefail
mkdir -p logs

MODEL_PATH="$1"
MODEL_NAME="$(basename "${MODEL_PATH}")"

echo "Starting evaluation pipeline for ${MODEL_NAME} at $(date)"
echo "MODEL_PATH=${MODEL_PATH}"

# Submit lm-eval job
LM_EVAL_JOBID=$(sbatch --parsable scripts/cscs/eval/lm_eval.sh "${MODEL_PATH}")
echo "Submitted lm-eval job: ${LM_EVAL_JOBID}"

# Submit babilong job
BABILONG_JOBID=$(sbatch --parsable scripts/cscs/eval/babilong.sh "${MODEL_PATH}")
echo "Submitted babilong job: ${BABILONG_JOBID}"

# Submit passkey job
PASSKEY_JOBID=$(sbatch --parsable scripts/cscs/eval/passkey.sh "${MODEL_PATH}")
echo "Submitted passkey job: ${PASSKEY_JOBID}"

# Submit perplexity job
PERPLEXITY_JOBID=$(sbatch --parsable scripts/cscs/eval/perplexity.sh "${MODEL_PATH}")
echo "Submitted perplexity job: ${PERPLEXITY_JOBID}"

echo "All evaluation jobs submitted at $(date)"
echo "  lm-eval:    ${LM_EVAL_JOBID}"
echo "  babilong:   ${BABILONG_JOBID}"
echo "  passkey:    ${PASSKEY_JOBID}"
echo "  perplexity: ${PERPLEXITY_JOBID}"
