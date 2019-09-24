#!/bin/bash

set -e

root_dir="$(dirname "${BASH_SOURCE[0]}")"
cd "$root_dir"

# Setup environment.
if [[ $(hostname --fqdn) = *"summit"* ]]; then
    cat > env.sh <<EOF
module load gcc/6.4.0
module load cuda/9.2.148
module load gsl
export CC=gcc
export CXX=g++

export USE_CUDA=${USE_CUDA:-0}
export USE_GASNET=${USE_GASNET:-1}
export CONDUIT=${CONDUIT:-ibv}

# for Numba
export CUDA_HOME=\$OLCF_CUDA_ROOT
EOF
elif [[ $(hostname) = "cori"* ]]; then
    cat > env.sh <<EOF
module unload PrgEnv-intel
module load PrgEnv-gnu
export CC=cc
export CXX=CC
export CRAYPE_LINK_TYPE=dynamic # allow dynamic linking

# disable Cori-specific Python environment
unset PYTHONSTARTUP

export USE_CUDA=${USE_CUDA:-0}
export USE_GASNET=${USE_GASNET:-1}
export CONDUIT=${CONDUIT:-aries}
EOF
elif [[ $(hostname) = "sapling" ]]; then
    cat > env.sh <<EOF
module load mpi/openmpi/3.1.3
module load gasnet/1.32.0-openmpi
module load cuda/8.0

export CC=gcc-6
export CXX=g++-6

export USE_CUDA=${USE_CUDA:-0}
export USE_GASNET=${USE_GASNET:-1}
export CONDUIT=${CONDUIT:-ibv}
EOF
elif [[ $(hostname) = "psbuild-"* ]]; then
    cat > env.sh <<EOF
export PATH="/opt/rh/devtoolset-7/root/usr/bin:$PATH"
export CC=gcc
export CXX=g++

export USE_CUDA=${USE_CUDA:-0}
export USE_GASNET=${USE_GASNET:-1}
export CONDUIT=${CONDUIT:-mpi}
EOF
else
    if [[ -z $CC || -z $CXX || ! $($CC --version) = *" 6."* || ! $($CXX --version) = *" 6."* ]]; then
        echo "GCC 6.x is required to build."
        echo "Please set CC/CXX to the right version and run again."
        echo
        echo "Note: This means machine auto-detection failed."
        exit 1
    fi
    cat > env.sh <<EOF
export CC="$CC"
export CXX="$CXX"

export USE_CUDA=${USE_CUDA:-0}
export USE_GASNET=${USE_GASNET:-0}
export CONDUIT=${CONDUIT:-mpi}
EOF
fi

cat >> env.sh <<EOF
export GASNET_ROOT="${GASNET_ROOT:-$PWD/gasnet/release}"

export LG_RT_DIR="${LG_RT_DIR:-$PWD/legion/runtime}"
export LEGION_DEBUG=1
export MAX_DIM=4

export PYVER=3.6

export LEGION_INSTALL_DIR="$PWD/install"
export PATH="\$LEGION_INSTALL_DIR/bin:\$PATH"
export LD_LIBRARY_PATH="\$LEGION_INSTALL_DIR/lib:\$LD_LIBRARY_PATH"
export PYTHONPATH="\$LEGION_INSTALL_DIR/lib/python\$PYVER/site-packages:\$PYTHONPATH"

export CONDA_ROOT="$PWD/conda"
export CONDA_ENV_DIR="\$CONDA_ROOT/envs/myenv"

export LCLS2_DIR="$PWD/lcls2"

export PATH="\$LCLS2_DIR/install/bin:\$PATH"
export PYTHONPATH="\$LCLS2_DIR/install/lib/python\$PYVER/site-packages:\$PYTHONPATH"

if [[ -d \$CONDA_ROOT ]]; then
  source "\$CONDA_ROOT/etc/profile.d/conda.sh"
  conda activate "\$CONDA_ENV_DIR"
fi
EOF

# Clean up any previous installs.
rm -rf conda
# rm -rf channels
# rm -rf relmanage
rm -rf lcls2

source env.sh

# Install Conda environment.
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-$(uname -p).sh -O conda-installer.sh
bash ./conda-installer.sh -b -p $CONDA_ROOT
rm conda-installer.sh
source $CONDA_ROOT/etc/profile.d/conda.sh

# conda install -y conda-build # Must be installed in root environment
PACKAGE_LIST=(
    # Stripped down from env_create.yaml:
    python=$PYVER
    cmake
    numpy
    cython
    matplotlib
    pytest
    mongodb
    pymongo
    curl
    rapidjson
    ipython
    requests
    mypy

    # Legion dependencies:
    cffi

    # pysingfel dependencies:
    numba
    h5py

    # Proxy app dependencies:
    scipy
)
if [[ $(hostname --fqdn) != *"summit"* && $(hostname) != "cori"* && $(hostname) != "sapling" ]]; then
    PACKAGE_LIST+=(
        mpi4py
    )
fi
if [[ $(hostname --fqdn) != *"summit"* ]]; then
    # no PPC64le version of this package
    PACKAGE_LIST+=(
        gsl
    )
fi
conda create -y -p "$CONDA_ENV_DIR" "${PACKAGE_LIST[@]}" -c defaults -c anaconda
# FIXME: Can't do this on Summit since not all the packages are available....
# git clone https://github.com/slac-lcls/relmanage.git
# sed s/PYTHONVER/$PYVER/ relmanage/env_create.yaml > temp_env_create.yaml
# conda env create -p "$CONDA_ENV_DIR" -f temp_env_create.yaml
conda activate "$CONDA_ENV_DIR"
# Other psana dependencies that live in the non-default channel.
conda install -y amityping -c lcls-ii
conda install -y bitstruct -c conda-forge

# Workaround for mpi4py not being built with the right MPI.
if [[ $(hostname --fqdn) = *"summit"* ]]; then
    CC=$OMPI_CC MPICC=mpicc pip install -v --no-binary mpi4py mpi4py
elif [[ $(hostname) = "cori"* ]]; then
    CC=gcc MPICC=cc pip install -v --no-binary mpi4py mpi4py
elif [[ $(hostname) = "sapling" ]]; then
    MPICC=mpicc pip install -v --no-binary mpi4py mpi4py
fi

# Install Legion.
# conda build relmanage/recipes/legion/ --output-folder channels/external/ --python $PYVER
# conda install -y legion -c file://`pwd`/channels/external --override-channels

if [[ $GASNET_ROOT == $PWD/gasnet/release ]]; then
    rm -rf gasnet
    git clone https://github.com/StanfordLegion/gasnet.git
    pushd gasnet
    make -j8
    popd
fi

if [[ $LG_RT_DIR == $PWD/legion/runtime ]]; then
    rm -rf legion
    rm -rf install
    git clone -b control_replication https://gitlab.com/StanfordLegion/legion.git
    ./reconfigure_legion.sh
    ./rebuild_legion.sh
fi

# Build psana.
git clone https://github.com/slac-lcls/lcls2.git $LCLS2_DIR
./psana_clean_build.sh

# Install phaseret.
./install_phaseret.sh

# Install pysingfel.
./install_pysingfel.sh

# Install cupy.
if [[ $(hostname) != "cori"* ]]; then
    ./install_cupy.sh
fi

echo
echo "Done. Please run 'source env.sh' to use this build."
