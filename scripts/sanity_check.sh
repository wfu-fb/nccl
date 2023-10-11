#!/bin/bash
BUILDDIR=${BUILDDIR:=/tmp/nccl-exp/genai-2.18.3/build}
CUDA_HOME=${CUDA_HOME:="/usr/local/cuda"}

echo "BUILDDIR=${BUILDDIR}"
echo "CUDA_HOME=${CUDA_HOME}"

# build example for sanity check
export BUILDDIR=$BUILDDIR
export NCCL_HOME=$BUILDDIR
export CUDA_HOME=$CUDA_HOME
cd examples && make clean && make all

# run example
NCCL_DEBUG=WARN LD_LIBRARY_PATH=$BUILDDIR/lib $BUILDDIR/examples/register
