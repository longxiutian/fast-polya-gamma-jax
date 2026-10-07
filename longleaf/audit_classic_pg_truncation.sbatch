#!/usr/bin/env bash
#SBATCH --job-name=nbe-pg-classic-audit
# Supply your allocation at submission: sbatch --account=YOUR_ACCOUNT ...
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --no-requeue
set -euo pipefail

: "${PG_DIAGNOSTIC_CODE:?}"
: "${PG_RUN_ROOT:?}"
: "${PG_DIAGNOSTIC_ROOT:?}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
python_bin="${PG_PYTHON:?Set PG_PYTHON to the cluster Python executable}"
"$python_bin" "$PG_DIAGNOSTIC_CODE/audit_classic_pg_truncation.py" \
  --run-root "$PG_RUN_ROOT" --index "$SLURM_ARRAY_TASK_ID" \
  --output-root "$PG_DIAGNOSTIC_ROOT"
