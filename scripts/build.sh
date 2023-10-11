#!/bin/bash
set -x

BUILDDIR=${BUILDDIR:=/tmp/nccl-exp/genai-2.18.3/build}
NVCC_ARCH=${NVCC_ARCH:="a100,h100"}
CUDA_HOME=${CUDA_HOME:="/usr/local/cuda"}
# TODO: automatically get hg id or git hash
DEV_SIGNITURE=${DEV_SIGNITURE:="none"}

echo "BUILDDIR=${BUILDDIR}"
echo "NVCC_ARCH=${NVCC_ARCH}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "DEV_SIGNITURE=${DEV_SIGNITURE}"

IFS=',' read -ra arch_array <<< "$NVCC_ARCH"
arch_gencode=""
for arch in "${arch_array[@]}"
do
    case "$arch" in
       "p100")
       arch_gencode="$arch_gencode -gencode=arch=compute_60,code=sm_60"
        ;;
       "v100")
       arch_gencode="$arch_gencode -gencode=arch=compute_70,code=sm_70"
        ;;
       "a100")
       arch_gencode="$arch_gencode -gencode=arch=compute_80,code=sm_80"
       ;;
       "h100")
        arch_gencode="$arch_gencode -gencode=arch=compute_90,code=sm_90"
       ;;
    esac
done

echo "NVCC_GENCODE=$arch_gencode"

# build libnccl
export BUILDDIR=$BUILDDIR
make src.clean && make -j src.build NVCC_GENCODE="$arch_gencode" CUDA_HOME="$CUDA_HOME" DEV_SIGNITURE="$DEV_SIGNITURE"
