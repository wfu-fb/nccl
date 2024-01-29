// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#pragma once

#include "DdaMemHandler.h"
#include "AlgoUtils.h"

namespace nccl {
namespace algorithms {

/* Base class to build collective-specific classes.  Contains common
 * infrastructure that is shared by two or more collectives. */
class AlgoManagerBase {
 public:
  AlgoManagerBase(ncclComm_t comm);
  ~AlgoManagerBase();

 protected:
  // we only support 2,4,8 ranks (single-node) for now
  static bool checkNumRanks(size_t numRanks);

  ncclComm_t comm_{nullptr};
  cudaDeviceProp devProp_;
  DdaMemHandler memHandler_;
  size_t maxBlocks_{0};

  // host memory
  DdaDeviceState* devStates_{nullptr};
  uintptr_t threadedBarrierFlag_{0};
  uintptr_t ipcBarrierFlag_{1};

  // device memory
  uintptr_t* threadedBarrierMbox_d_{nullptr};
  uintptr_t* ipcBarrierMbox_d_{nullptr};
  void* tmpbuff_d_{nullptr};
  DdaDeviceState* devStates_d_{nullptr};
};

} // namespace algorithms
} // namespace nccl
