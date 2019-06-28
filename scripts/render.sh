#!/bin/bash

set -e

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$root_dir"/../setup/env.sh

OUT_DIR=$MEMBERWORK/chm137/minifel_output

pushd $OUT_DIR
$root_dir/check_output.py
popd
