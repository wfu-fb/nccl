// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <stdio.h>
#include <cstddef>
#include "CtranAlgoDev.h"
#include "CtranGpeDev.h"
#include "CtranGpeKernel.h"
#include "ctran_kernel.h"
#include "nccl.h"

template <typename T>
__device__ __forceinline__ void ctranKernSend(
    const T* sendbuff,
    size_t count,
    CtranAlgoDeviceState* devState,
    int groupIdx,
    int ngroups) {
  int localRank = devState->localRank;
  int localRanks = devState->localRanks;
  size_t bufSize = devState->bufSize;

  for (int r = 1; r < localRanks; r++) {
    // Ensure each rank sends to different peer at a time to avoid alltoone P2P
    // write congestion. For example, with localRanks = 4, the following
    // schedule is used:
    // - Round0:
    // rank0: s(1)r(3); rank1: s(2)r(0); rank2: s(3)r(1); rank3: s(0)r(2)
    // - Round1:
    // rank0: s(2)r(2); rank1: s(3)r(3); rank2: s(0)r(0); rank3: s(1)r(1)
    // - Round2:
    // rank0: s(3)r(1); rank1: s(0)r(2); rank2: s(1)r(3); rank3: s(2)r(0)
    int sendPeer = (localRank + r) % localRanks;
    size_t displ = count * devState->localRankToRank[sendPeer];

    // get shared buffer and states
    CtranAlgoDeviceBufState* bufState =
        devState->allPeerToBufStatesMap[sendPeer][localRank];
    void* buf = devState->allPeerToBufsMap[sendPeer][localRank];
    const T* sendPtr = sendbuff + displ;

    if (canCopy16(sendPtr, count)) {
      multiStepsSend<uint4>(
          reinterpret_cast<const uint4*>(sendPtr),
          count * sizeof(T) / sizeof(uint4),
          reinterpret_cast<uint4*>(buf),
          bufState,
          bufSize / sizeof(uint4),
          groupIdx,
          ngroups);
    } else {
      multiStepsSend<T>(
          sendPtr,
          count,
          reinterpret_cast<T*>(buf),
          bufState,
          bufSize / sizeof(T),
          groupIdx,
          ngroups);
    }
  }
}

template <typename T>
__device__ __forceinline__ void ctranKernRecv(
    T* recvbuff,
    size_t count,
    CtranAlgoDeviceState* devState,
    int groupIdx,
    int ngroups) {
  int localRank = devState->localRank;
  int localRanks = devState->localRanks;
  size_t bufSize = devState->bufSize;

  for (int r = 1; r < localRanks; r++) {
    // Ensure each rank sends to different peer at a time to avoid alltoone P2P
    // write congestion. For example, with localRanks = 4, the following
    // schedule is used:
    // - Round0:
    // rank0: s(1)r(3); rank1: s(2)r(0); rank2: s(3)r(1); rank3: s(0)r(2)
    // - Round1:
    // rank0: s(2)r(2); rank1: s(3)r(3); rank2: s(0)r(0); rank3: s(1)r(1)
    // - Round2:
    // rank0: s(3)r(1); rank1: s(0)r(2); rank2: s(1)r(3); rank3: s(2)r(0)

    int recvPeer = (localRank + localRanks - r) % localRanks;
    size_t displ = count * devState->localRankToRank[recvPeer];

    // get shared buffer and states
    CtranAlgoDeviceBufState* bufState =
        devState->allPeerToBufStatesMap[localRank][recvPeer];
    void* buf = devState->allPeerToBufsMap[localRank][recvPeer];
    T* recvPtr = recvbuff + displ;

    if (canCopy16(recvPtr, count)) {
      multiStepsRecv<uint4>(
          reinterpret_cast<uint4*>(recvPtr),
          count * sizeof(T) / sizeof(uint4),
          reinterpret_cast<const uint4*>(buf),
          bufState,
          bufSize / sizeof(uint4),
          groupIdx,
          ngroups);
    } else {
      multiStepsRecv<T>(
          recvPtr,
          count,
          reinterpret_cast<const T*>(buf),
          bufState,
          bufSize / sizeof(T),
          groupIdx,
          ngroups);
    }
  }
}

template <typename T>
__device__ __forceinline__ void ctranKernSelf(
    const T* sendbuff,
    T* recvbuff,
    size_t count,
    CtranAlgoDeviceState* devState) {
  if (sendbuff != recvbuff) {
    if (canCopy16(sendbuff, recvbuff, count)) {
      copy<uint4>(
          reinterpret_cast<uint4*>(recvbuff),
          reinterpret_cast<const uint4*>(sendbuff),
          count * sizeof(T) / sizeof(uint4));
    } else {
      copy<T>(recvbuff, sendbuff, count);
    }
  }
}

enum { GROUP_SEND, GROUP_RECV };

template <typename T>
__global__ void ncclKernelAllToAll(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelAllToAllArgs args) {
  const int gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  if (flag && gtIdx == 0) {
    ncclKernelStartGpe(flag);
  }

  const T* sendbuff = reinterpret_cast<const T*>(args.sendbuff);
  T* recvbuff = reinterpret_cast<T*>(args.recvbuff);
  size_t count = args.count;

  // All blocks are involved in self D2D copy
  int localRank = devState->localRank;
  size_t selfDisp = count * devState->localRankToRank[localRank];
  ctranKernSelf<T>(sendbuff + selfDisp, recvbuff + selfDisp, count, devState);

  // Partition blocks into a set of send groups and a set of receive groups
  // Let even blocks handle NVL sends, and odd blocks handle NVL receives,
  // and assign groupIdx 0, 1, 2... for block{0,2,4...}@sender and
  // block{1,3,5...}@receiver. The same groupIdx on sender and receiver
  // coordinates to finish a pair of send-receive.
  const int ngroups = gridDim.x / 2;
  const int groupIdx = blockIdx.x / 2;
  const bool groupType = blockIdx.x % 2 == 0 ? GROUP_SEND : GROUP_RECV;

  if (groupType == GROUP_RECV) {
    ctranKernRecv(recvbuff, count, devState, groupIdx, ngroups);
  } else {
    ctranKernSend(sendbuff, count, devState, groupIdx, ngroups);
  }

  if (flag && gtIdx == 0) {
    ncclKernelWaitGpeTerminate(flag);
  }
}

#define DECL_ALLTOALL_KERN(T)                     \
  template __global__ void ncclKernelAllToAll<T>( \
      int* flag, CtranAlgoDeviceState* devState, CtranKernelAllToAllArgs args)

DECL_ALLTOALL_KERN(int8_t);
DECL_ALLTOALL_KERN(uint8_t);
DECL_ALLTOALL_KERN(int32_t);
DECL_ALLTOALL_KERN(uint32_t);
DECL_ALLTOALL_KERN(int64_t);
DECL_ALLTOALL_KERN(uint64_t);
DECL_ALLTOALL_KERN(half);
DECL_ALLTOALL_KERN(float);
DECL_ALLTOALL_KERN(double);
#if defined(__CUDA_BF16_TYPES_EXIST__)
DECL_ALLTOALL_KERN(__nv_bfloat16);
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
DECL_ALLTOALL_KERN(__nv_fp8_e4m3);
DECL_ALLTOALL_KERN(__nv_fp8_e5m2);
#endif
