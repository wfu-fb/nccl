// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#include "AlgoManager.h"

#include "debug.h"

#include <cassert>

namespace nccl {
namespace algorithms {

/**
 * per communicator per rank Algorithm Manager that
 * - manages all the available algorithm instances for a given collective
 * - selects an optimal algorithm based on the input and environments
 */
AlgoManager::AlgoManager(ncclComm_t comm) : comm_(comm) {
  INFO(NCCL_COLL, "AlgoManager initialized.");
}

std::unique_ptr<AllReduceAlgo> AlgoManager::getAllReduceAlgo(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  return nullptr;
}

} // namespace algorithms
} // namespace nccl
