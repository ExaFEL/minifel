#!/bin/bash

set -e

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$root_dir"/../native_kernels

rm -rf build
"$root_dir"/dirty_build.sh
