// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#include "AlgoInit.h"

namespace nccl {
namespace algorithms {

ncclResult_t algoInit(ncclComm_t comm) {
  // initiate AlgoManager
  comm->algoMgr = std::unique_ptr<nccl::algorithms::AlgoManager>(
      new nccl::algorithms::AlgoManager(comm));
  return ncclSuccess;
}

} // namespace algorithms
} // namespace nccl
