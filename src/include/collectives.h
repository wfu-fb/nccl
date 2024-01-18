/*************************************************************************
 * Copyright (c) 2017-2022, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_COLLECTIVES_H_
#define NCCL_COLLECTIVES_H_

#include "nccl.h"

#include <stdint.h>

// CHUNKSIZE must be a multiple of SLICESIZE
#define ALLREDUCE_SLICESTEPS (NCCL_STEPS/4)
#define ALLREDUCE_CHUNKSTEPS (NCCL_STEPS/2)
#define ALLGATHER_SLICESTEPS (NCCL_STEPS/4)
#define ALLGATHER_CHUNKSTEPS (NCCL_STEPS/2)
#define REDUCESCATTER_SLICESTEPS (NCCL_STEPS/4)
#define REDUCESCATTER_CHUNKSTEPS (NCCL_STEPS/2)
#define BROADCAST_SLICESTEPS 1
#define BROADCAST_CHUNKSTEPS 1
#define REDUCE_SLICESTEPS 1
#define REDUCE_CHUNKSTEPS 1
#define NCCL_MAX_SLICE_PER_CHUNK 2  // max value for CHUNKSTEPS/SLICESTEPS, must accord with above

inline int ncclTypeSize(ncclDataType_t type) {
  switch (type) {
  case ncclInt8:
  case ncclUint8:
    return 1;
  case ncclFloat16:
  #if defined(__CUDA_BF16_TYPES_EXIST__)
  case ncclBfloat16:
  #endif
    return 2;
  case ncclInt32:
  case ncclUint32:
  case ncclFloat32:
    return 4;
  case ncclInt64:
  case ncclUint64:
  case ncclFloat64:
    return 8;
  default:
    return -1;
  }
}

struct DdaDeviceState {
  uintptr_t* threadedBarrierMbox;
  uintptr_t* ipcBarrierMbox;
  void* tmpbuff;
};

// DDA kernels
template <typename T, uint32_t NRANKS>
__global__ void ncclKernel_AllReduce_DDA_Flat(
  uintptr_t barrierFlag, DdaDeviceState* devStates,
  int rank, const T* sendbuff, T* recvbuff, size_t count);
template <typename T, uint32_t NRANKS>
__global__ void ncclKernel_AllReduce_DDA_Tree(
  uintptr_t barrierFlag, DdaDeviceState* devStates,
  int rank, const T* sendbuff, T* recvbuff, size_t count);
template <typename T, uint32_t NRANKS>
__global__ void ncclKernel_AllReduce_DDA_Flat_ipc(
  uintptr_t barrierFlag, DdaDeviceState* devStates,
  int rank, T* recvbuff, size_t count);
template <typename T, uint32_t NRANKS>
__global__ void ncclKernel_AllReduce_DDA_Tree_ipc(
  uintptr_t barrierFlag, DdaDeviceState* devStates,
  int rank, T* recvbuff, size_t count);
template <typename T, uint32_t NRANKS>
__global__ void ncclKernel_AllReduce_DDA_ScatGat_ipc(
  uintptr_t barrierFlag, DdaDeviceState* devStates,
  int rank, T* sendbuff, T* recvbuff, size_t count);

#endif
