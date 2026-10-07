#!/usr/bin/env bash
#SBATCH --job-name=nbe-pg-classic-verify
# Supply your allocation at submission: sbatch --account=YOUR_ACCOUNT ...
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --no-requeue
set -euo pipefail

: "$PG_RELEASE"
: "$NBE_CODE_ROOT"
: "$PG_RUN_ROOT"
export JAX_PLATFORMS=cpu
export PYTHONPATH="$PG_RELEASE/src:$NBE_CODE_ROOT/src"
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
root="$PG_RUN_ROOT/$config/chain-$chain"
"$python_bin" "$PG_RELEASE/scripts/verify_classic_hb_teacher.py" \
  --chain-root "$root" --write-marker
