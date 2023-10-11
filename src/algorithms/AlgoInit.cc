// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#include "AlgoInit.h"

namespace nccl {
namespace algorithms {

ncclResult_t algoInit(ncclComm_t comm, bool forceInit) {
  const char* allReduceAlgoStr = getenv("NCCL_ALLREDUCE_ALGO");
  if ((allReduceAlgoStr == nullptr || strcmp(allReduceAlgoStr, "dda2")) &&
      (!forceInit)) {
    // NCCL_ALLREDUCE_ALGO != dda2 and !forceInit, skip initialization
    return ncclSuccess;
  }

  // initiate AlgoManager
  comm->algoMgr = std::unique_ptr<nccl::algorithms::AlgoManager>(
      new nccl::algorithms::AlgoManager(comm));
  return ncclSuccess;
}

} // namespace algorithms
} // namespace nccl
