#!/bin/bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
source "$root_dir"/env.sh

pushd "$root_dir"

rm -rf cupy
git clone https://github.com/cupy/cupy.git
# git -C cupy reset --hard 7dca63a17d1f517bcf5595ddcd32a0342d6a95b5 # https://github.com/cupy/cupy/issues/2433

conda install -y fastrlock

pushd cupy

pip install --no-cache-dir .

popd

popd
