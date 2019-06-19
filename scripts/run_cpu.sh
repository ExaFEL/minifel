#!/bin/bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
source "$root_dir"/../setup/env.sh

cd "$root_dir"/..

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$root_dir/native_kernels/build:$CONDA_ENV_DIR/lib"
export PS_PARALLEL=legion

export KERNEL_KIND=sum
export LIMIT=10

export DATA_DIR="${DATA_DIR:-/reg/neh/home/dujardin/minifel/data/marc}"
if [[ ! -d $DATA_DIR ]]; then
    echo "DATA_DIR is not set or does not exist. Please check it and rerun."
    false
fi

export OUT_DIR="${OUT_DIR:-/reg/neh/home/dujardin/minifel/output}"
if [[ ! -d $DATA_DIR ]]; then
    echo "OUT_DIR is not set or does not exist. Please check it and rerun."
    false
fi

# legion_python user -ll:py 1 -ll:cpu 1 -level announce=2
mpirun -n 2 legion_python user -ll:py 1 -ll:cpu 1 -level announce=2
