// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#ifndef CTRAN_ALGO_H_
#define CTRAN_ALGO_H_

#include <vector>
#include "CtranAlgoDev.h"
#include "nccl.h"

struct ncclComm;

#define LOCAL_RANK_TO_DEV_REGION_POS(localRank, ownerLocalRank) \
  (localRank < ownerLocalRank ? localRank : localRank - 1)

class CtranAlgo {
 public:
  CtranAlgo(ncclComm* comm);
  ~CtranAlgo();

  // Release inter-process shared resource collectively (e.g., unmap IPC handle
  // before local cudaFree). Used in ncclCommAbort->ctranAbort before bootstrap
  // socket is closed.
  void releaseSharedResource();

  // Most Ctran algorithms are only for multi-process comm. Check this function
  // before switching to any ctran algorithm.
  bool isThreadedComm(ncclComm* comm);

  // Device buffer to store all states of
  // shared device buffers and comm info, accessed by kernels.
  CtranAlgoDeviceState* devState_d;

 private:
  class SharedResource;

  ncclResult_t initDevState();
  ncclResult_t destroyDevState();

  ncclComm* comm_{nullptr};
  SharedResource* sharedRes_{nullptr};
};

class CtranAlgo::SharedResource {
 public:
  SharedResource(ncclComm* comm);
  ~SharedResource();

  std::vector<void*> mappedDevShmPtrs; // pointer to mapped device memory
                                       // regions of remote peers

 private:
  ncclResult_t barrier();
  ncclComm* comm_{nullptr};
  void* devShmPtr_{nullptr}; // pointer to my local device memory region
};

#endif
