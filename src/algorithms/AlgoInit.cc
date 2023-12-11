// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#include "AlgoInit.h"
#include "nccl_cvars.h"

namespace nccl {
namespace algorithms {

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_ALLREDUCE_ALGO2
   type        : enum
   default     : orig
   choices     : orig, dda
   description : |-
     The algorithm to use for Allreduce communication
     orig - Copy-based algorithm
     dda - Direct Data Access algorithms

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

ncclResult_t algoInit(ncclComm_t comm, bool forceInit) {
  if ((NCCL_ALLREDUCE_ALGO2 != NCCL_ALLREDUCE_ALGO2::dda) && (!forceInit)) {
    // NCCL_ALLREDUCE_ALGO2 != dda and !forceInit, skip initialization
     return ncclSuccess;
   }

  // initiate AlgoManager
  comm->algoMgr = std::unique_ptr<nccl::algorithms::AlgoManager>(
      new nccl::algorithms::AlgoManager(comm));
  return ncclSuccess;
}

ncclResult_t algoDestroy(ncclComm_t comm) {
  if (comm->algoMgr) {
    comm->algoMgr.reset();
  }
  return ncclSuccess;
}

} // namespace algorithms
} // namespace nccl
