#!/bin/bash

if [[ -d $DATA_DIR ]]; then
    dest=$DATA_DIR
else
    if [[ $(hostname --fqdn) = *"summit"* ]]; then
        dest=$MEMBERWORK/chm137/align_data
    elif [[ $(hostname) = "cori"* ]]; then
        dest=$SCRATCH/align_data
    elif [[ $(hostname) = "sapling"* ]]; then
        dest=/scratch/oldhome/$(whoami)/align_data
    else
        echo "Unable to auto-detect the machine"
        exit 1
    fi
fi

mkdir -p $dest

rsync -rzP --exclude '*.hdf5' ${PSUSER:-$USER}@psexport.slac.stanford.edu:/reg/d/psdm/xpp/xpptut15/scratch/dujardin/minifel/01/ $dest
