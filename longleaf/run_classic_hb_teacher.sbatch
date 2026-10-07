#!/usr/bin/env bash
#SBATCH --job-name=nbe-pg-classic
# Supply your allocation at submission: sbatch --account=YOUR_ACCOUNT ...
#SBATCH --qos=gpu_access
#SBATCH --partition=l40-gpu
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --no-requeue
set -euo pipefail

: "${PG_CODE_ROOT:?}"
: "${NBE_CODE_ROOT:?}"
: "${PG_PREP_S12:?}"
: "${PG_PREP_S34:?}"
: "${PG_RUN_ROOT:?}"
: "${PG_MODE:?}"
cd "$PG_CODE_ROOT"
export PYTHONPATH="$PG_CODE_ROOT/src:$NBE_CODE_ROOT/src"
export JAX_ENABLE_X64=true
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_DEFAULT_MATMUL_PRECISION=highest
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1

task="${SLURM_ARRAY_TASK_ID:-0}"
index=$((task / 2))
chain=$((task % 2))
case "$PG_MODE" in
  smoke) warmup=20; samples=20; chunk=10; global_stride=1; person_stride=10 ;;
  pilot) warmup=100; samples=1000; chunk=100; global_stride=10; person_stride=100 ;;
  production)
    warmup=20000; samples=1000000; chunk=1000
    global_stride=20; person_stride=1000 ;;
  *) echo "Unknown PG_MODE: $PG_MODE" >&2; exit 2 ;;
esac

cache="$PG_RUN_ROOT/cache/task-$task"
mkdir -p "$cache"
export XDG_CACHE_HOME="$cache"
export TMPDIR="$cache/tmp"
export CUDA_CACHE_PATH="$cache/cuda"
mkdir -p "$TMPDIR" "$CUDA_CACHE_PATH"
python_bin="${PG_PYTHON:?Set PG_PYTHON to the cluster Python executable}"
test -x "$python_bin"
test -f "$PG_CODE_ROOT/src/fast_polya_gamma_jax/classic_hb_gibbs.py"
test -f "$NBE_CODE_ROOT/src/mengyao_nbe/refinement/micro_panel_teacher.py"
scontrol show job --oneliner "$SLURM_JOB_ID" > "$cache/job-allocation.txt"
nvidia-smi --query-gpu=timestamp,uuid,utilization.gpu,utilization.memory,memory.used,memory.total \
  --format=csv --loop=30 > "$cache/gpu-${SLURM_JOB_ID}.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
/usr/bin/time -v -o "$cache/time-${SLURM_JOB_ID}.txt" \
  "$python_bin" "$PG_CODE_ROOT/scripts/run_classic_hb_teacher.py" \
    --prep-s12 "$PG_PREP_S12" --prep-s34 "$PG_PREP_S34" \
    --index "$index" --chain "$chain" --run-root "$PG_RUN_ROOT" \
    --mode "$PG_MODE" --warmup "$warmup" --samples "$samples" \
    --chunk-size "$chunk" --global-archive-stride "$global_stride" \
    --person-archive-stride "$person_stride" --num-terms "${PG_NUM_TERMS:-8}"
