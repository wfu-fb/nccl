// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_ALGO_DEV_H_
#define CTRAN_ALGO_DEV_H_

#include <cstddef>

#define CTRAN_ALGO_STEP_RESET (-1)
#define CTRAN_ALGO_MAX_THREAD_BLOCKS (64)
#define CTRAN_MAX_NVL_PEERS (8)

struct alignas(16) CtranAlgoDeviceBufState {
  // Separate flag per thread block to coordinate with remote ranks
  // independently
  volatile int stepOnSameBlockIdx[CTRAN_ALGO_MAX_THREAD_BLOCKS];
};

struct alignas(16) CtranAlgoDeviceState {
  // Shared buffers for intra-node inter-process communication.
  // Both bufState and buf are pointers to device memory.
  CtranAlgoDeviceBufState* allPeerToBufStatesMap[CTRAN_MAX_NVL_PEERS]
                                                [CTRAN_MAX_NVL_PEERS];
  void* allPeerToBufsMap[CTRAN_MAX_NVL_PEERS][CTRAN_MAX_NVL_PEERS];

  // Comm info copied from ncclComm
  int localRankToRank[CTRAN_MAX_NVL_PEERS];
  size_t bufSize;
  int localRank;
  int localRanks;
};

#endif
