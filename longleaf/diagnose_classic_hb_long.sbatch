#!/usr/bin/env bash
#SBATCH --job-name=nbe-pg-classic-diag
# Supply your allocation at submission: sbatch --account=YOUR_ACCOUNT ...
#SBATCH --partition=general
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --no-requeue
set -euo pipefail

: "$PG_DIAGNOSTIC_CODE"
: "$PG_RUN_ROOT"
: "$PG_DIAGNOSTIC_ROOT"
export JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
python_bin="${PG_PYTHON:?Set PG_PYTHON to the cluster Python executable}"
"$python_bin" "$PG_DIAGNOSTIC_CODE/diagnose_classic_hb_long.py" \
  --run-root "$PG_RUN_ROOT" --index "$SLURM_ARRAY_TASK_ID" \
  --output-root "$PG_DIAGNOSTIC_ROOT"
