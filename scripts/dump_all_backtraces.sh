#!/bin/bash

set -x

root_dir="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

job="$LSB_JOBID"
scratch_dir="$MEMBERWORK/chm137/minifel_backtrace_$job"
mkdir -p "$scratch_dir"

i=0
hosts="$(bjobs -o exec_host $job | tail -n -1 | tr ':' '\n' | sort -u | tr '\n' ' ')"
for host in $hosts; do
    ssh $host bash "$root_dir/dump_node_backtraces.sh" "$scratch_dir" &
    let i++
    if [[ $(( i % 200 )) == 0 ]]; then
        wait
    fi
done

wait
