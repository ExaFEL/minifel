#!/bin/bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
source "$root_dir"/env.sh

pushd "$root_dir"

rm -rf cupy
git clone https://github.com/cupy/cupy.git

conda install -y fastrlock

pushd cupy

pip install --no-cache-dir .

popd

popd
