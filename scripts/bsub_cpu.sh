#!/bin/bash
#BSUB -P CHM137
#BSUB -W 0:30
#BSUB -nnodes 1
#BSUB -alloc_flags NVME
#BSUB -o lsf-%J.out
#BSUB -e lsf-%J.err
#BSUB -N

root_dir="$PWD"

source "$root_dir"/setup/env.sh

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$root_dir/native_kernels/build"
export PYTHONPATH="$PYTHONPATH:$root_dir"
export PS_PARALLEL=legion

export DATA_DIR=$MEMBERWORK/chm137/align_data

export OUT_DIR=$MEMBERWORK/chm137/minifel_output
mkdir -p $OUT_DIR

export LIMIT=10

export IBV_FORK_SAFE=1 # workaround for https://upc-bugs.lbl.gov/bugzilla/show_bug.cgi?id=3908

export all_proxy=socks://proxy.ccs.ornl.gov:3128/
export ftp_proxy=ftp://proxy.ccs.ornl.gov:3128/
export http_proxy=http://proxy.ccs.ornl.gov:3128/
export https_proxy=https://proxy.ccs.ornl.gov:3128/
export no_proxy='localhost,127.0.0.0/8,*.ccs.ornl.gov,*.ncrc.gov'

nodes=$(( ( LSB_MAX_NUM_PROCESSORS - 1 ) / 42 ))

# CuPy tries to store its cache in ~/.cupy
# export HOME=$MEMBERWORK/chm137/run
# mkdir -p $HOME
export CUPY_CACHE_DIR=/mnt/bb/$USER/.cupy/kernel_cache

time jsrun -n $(( nodes * 2 )) --rs_per_host 2 --tasks_per_rs 1 --cpu_per_rs 21 --gpu_per_rs 1 --bind rs "$root_dir"/scripts/pick_hcas.py legion_python user -ll:py 1 -ll:cpu 1 -ll:csize 8192 -level announce=2
