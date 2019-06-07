#!/bin/bash

set -e

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root_dir"/../native_kernels

source "$root_dir"/../setup/env.sh

mkdir -p build
cd build
cmake -DCMAKE_PREFIX_PATH="$LEGION_INSTALL_DIR" ..
make -j8
