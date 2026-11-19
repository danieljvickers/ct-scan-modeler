#!/usr/bin/env bash
# Georgia Tech wingtip GPU nodes, 4 x NVIDIA A100 80 GB
# aliases: w

export NVHPC_ROOT=/fastscratch/dvickers6/nvhpc/Linux_x86_64/25.11

export PATH="$NVHPC_ROOT/compilers/bin:$PATH"
export LD_LIBRARY_PATH="$NVHPC_ROOT/compilers/lib:$LD_LIBRARY_PATH"

export PATH="$NVHPC_ROOT/comm_libs/mpi/bin:$PATH"
export LD_LIBRARY_PATH="$NVHPC_ROOT/comm_libs/mpi/lib:$LD_LIBRARY_PATH"

export PATH="/usr/lib/gcc/x86_64-linux-gnu/13:$PATH"
export NVHPC_GNU_INSTALL_ROOT=/usr/lib/gcc/x86_64-linux-gnu/13/

export CC=nvc
export CXX=nvc++
export FC=nvfortran

export TMPDIR=$HOME/insight_temp

export CTM_CUDA_ARCHITECTURES=80  # A100

# no InfiniBand: disable HPC-X collectives and the PMIx shared memory store, which fail without it
export OMPI_MCA_coll_hcoll_enable=0
export PMIX_MCA_gds=hash

# CUDA aware all to all hangs on the 1.5M cell shedding case, stage MPI through host memory until that is understood
export CTM_CUDA_AWARE_MPI=0
