#!/usr/bin/env bash
#SBATCH --job-name=nbe-pg-mix-diag
# Supply your allocation at submission: sbatch --account=YOUR_ACCOUNT ...
#SBATCH --partition=general
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --no-requeue
set -euo pipefail

: "$MIX_DIAGNOSTIC_CODE"
: "$MIX_RUN_ROOT"
: "$MIX_DIAGNOSTIC_ROOT"
export JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
python_bin="${PG_PYTHON:?Set PG_PYTHON to the cluster Python executable}"
"$python_bin" "$MIX_DIAGNOSTIC_CODE/diagnose_sparse_mixture_long.py" \
  --run-root "$MIX_RUN_ROOT" --index "$SLURM_ARRAY_TASK_ID" \
  --output-root "$MIX_DIAGNOSTIC_ROOT"
