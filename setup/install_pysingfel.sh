#!/bin/bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
source "$root_dir"/env.sh

pushd "$root_dir"/pysingfel

conda install -y numba
pip install -e .

popd
