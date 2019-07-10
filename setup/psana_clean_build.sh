#!/bin/bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
source "$root_dir"/env.sh

pushd $LCLS2_DIR
git clean -fxd
popd

"$root_dir"/psana_dirty_build.sh
