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
    KernelP2pElem* sendElemsList,
    CtranAlgoDeviceState* devState,
    int groupIdx,
    int ngroups) {
  int localRank = devState->localRank;
  size_t bufSize = devState->bufSize;

  // Host algorithm already schedules send and receives with different peer to
  // avoid P2P congistion. Thus, kernel just runs following the list sequence
  KernelP2pElem* elem = sendElemsList;
  while (elem != nullptr) {
    int sendPeer = elem->peerRank;
    size_t count = elem->count;
    size_t displ = elem->displ;

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
    elem = elem->next;
  }
}

template <typename T>
__device__ __forceinline__ void ctranKernRecv(
    T* recvbuff,
    KernelP2pElem* recvElemsList,
    CtranAlgoDeviceState* devState,
    int groupIdx,
    int ngroups) {
  int localRank = devState->localRank;
  size_t bufSize = devState->bufSize;

  // Host algorithm already schedules send and receives with different peer to
  // avoid P2P congistion. Thus, kernel just runs following the list sequence
  KernelP2pElem* elem = recvElemsList;
  while (elem != nullptr) {
    int recvPeer = elem->peerRank;
    size_t count = elem->count;
    size_t displ = elem->displ;

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
    elem = elem->next;
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

// Reset inuse flag of all elements in the list to allow the host pool to
// reclaim them.
// Note that the last resetStep called in multiStepsSend|Recv ensures all
// threads in the block have finished the last step; thus, safe to release all
// elem objects back to host pool;
__device__ __forceinline__ void ctranKernCompleteElem(
    KernelP2pElem* elemsList,
    int groupIdx) {
  KernelP2pElem* elem = elemsList;
  while (elem != nullptr) {
    elem->inuse[groupIdx] = false;
    elem = elem->next;
  }
}

enum { GROUP_SEND, GROUP_RECV };

template <typename T>
__global__ void ncclKernelAllToAllv(
    int* flag,
    CtranAlgoDeviceState* devState,
    CtranKernelAllToAllvArgs args) {
  const int gtIdx = blockDim.x * blockIdx.x + threadIdx.x;

  if (flag && gtIdx == 0) {
    ncclKernelStartGpe(flag);
  }

  const T* sendbuff = reinterpret_cast<const T*>(args.sendbuff);
  T* recvbuff = reinterpret_cast<T*>(args.recvbuff);
  size_t selfCount = args.selfCount;
  size_t selfSendDispl = args.selfSendDispl;
  size_t selfRecvDispl = args.selfRecvDispl;
  KernelP2pElem* sendElemsList = args.sendElemsList;
  KernelP2pElem* recvElemsList = args.recvElemsList;

  // All blocks are involved in self D2D copy
  ctranKernSelf<T>(
      sendbuff + selfSendDispl, recvbuff + selfRecvDispl, selfCount, devState);

  // Partition blocks into a set of send groups and a set of receive groups
  // Let even blocks handle NVL sends, and odd blocks handle NVL receives,
  // and assign groupIdx 0, 1, 2... for block{0,2,4...}@sender and
  // block{1,3,5...}@receiver. The same groupIdx on sender and receiver
  // coordinates to finish a pair of send-receive.
  const int ngroups = gridDim.x / 2;
  const int groupIdx = blockIdx.x / 2;
  const bool groupType = blockIdx.x % 2 == 0 ? GROUP_SEND : GROUP_RECV;

  if (groupType == GROUP_RECV) {
    ctranKernRecv(recvbuff, recvElemsList, devState, groupIdx, ngroups);
    if (threadIdx.x == 0) {
      ctranKernCompleteElem(recvElemsList, groupIdx);
    }
  } else {
    ctranKernSend(sendbuff, sendElemsList, devState, groupIdx, ngroups);
    if (threadIdx.x == 0) {
      ctranKernCompleteElem(sendElemsList, groupIdx);
    }
  }

  if (flag && gtIdx == 0) {
    ncclKernelWaitGpeTerminate(flag);
  }
}

#define DECL_ALLTOALLV_KERN(T)                     \
  template __global__ void ncclKernelAllToAllv<T>( \
      int* flag, CtranAlgoDeviceState* devState, CtranKernelAllToAllvArgs args)


DECL_ALLTOALLV_KERN(int8_t);
DECL_ALLTOALLV_KERN(uint8_t);
DECL_ALLTOALLV_KERN(int32_t);
DECL_ALLTOALLV_KERN(uint32_t);
DECL_ALLTOALLV_KERN(int64_t);
DECL_ALLTOALLV_KERN(uint64_t);
DECL_ALLTOALLV_KERN(half);
DECL_ALLTOALLV_KERN(float);
DECL_ALLTOALLV_KERN(double);
#if defined(__CUDA_BF16_TYPES_EXIST__)
DECL_ALLTOALLV_KERN(__nv_bfloat16);
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
DECL_ALLTOALLV_KERN(__nv_fp8_e4m3);
DECL_ALLTOALLV_KERN(__nv_fp8_e5m2);
#endif
