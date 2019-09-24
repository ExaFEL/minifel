#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --qos=debug # regular
#SBATCH --constraint=haswell
#SBATCH --mail-type=ALL
#SBATCH --account=m2859

root_dir="$PWD"

source "$root_dir"/../setup/env.sh

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$root_dir/../native_kernels/build"
export PYTHONPATH="$PYTHONPATH:$root_dir/.."
export PS_PARALLEL=legion

export DATA_DIR=$SCRATCH/align_data

export OUT_DIR=$SCRATCH/minifel_output
mkdir -p $OUT_DIR

export LIMIT=10

nodes=$SLURM_JOB_NUM_NODES

export OMP_NUM_THREADS=1 # attempt to disable NumPy use of threads

cores=10
srun -n $(( nodes * cores )) -N $nodes --cpus-per-task $(( 64 / cores )) --cpu_bind cores legion_python main -ll:py 1 -ll:cpu 0 -ll:csize 8192
