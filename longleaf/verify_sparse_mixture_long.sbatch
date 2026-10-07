#!/usr/bin/env bash
#SBATCH --job-name=nbe-pg-mix-verify
# Supply your allocation at submission: sbatch --account=YOUR_ACCOUNT ...
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --no-requeue
set -euo pipefail

: "$MIX_RELEASE"
: "$MIX_RUN_ROOT"
export JAX_PLATFORMS=cpu
export PYTHONPATH="$MIX_RELEASE/src"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
python_bin="${PG_PYTHON:?Set PG_PYTHON to the cluster Python executable}"

task="$SLURM_ARRAY_TASK_ID"
split=$((task / 2))
chain=$((task % 2))
case "$split" in
  0) config=global-exposure80-seed2026090902 ;;
  1) config=within-campaign-last20pct-weeks-v1 ;;
  2) config=within-campaign-one-ad-seed2026091103 ;;
  3) config=hybrid-one-ad-plus-known-exposure20-seed2026091504 ;;
  *) exit 2 ;;
esac
root="$MIX_RUN_ROOT/$config/chain-$chain"
"$python_bin" "$MIX_RELEASE/scripts/verify_sparse_mixture_teacher.py" \
  --chain-root "$root" --write-marker
