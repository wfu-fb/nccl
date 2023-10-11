// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#include "AlgoManager.h"

#include "DdaThreadedData.h"
#include "argcheck.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"

#include <cassert>

namespace nccl {
namespace algorithms {

/**
 * per communicator per rank Algorithm Manager that
 * - manages all the available algorithm instances for a given collective
 * - selects an optimal algorithm based on the input and environments
 */
AlgoManager::AlgoManager(ncclComm_t comm) : comm_(comm), memHandler_(comm) {
  // register rank
  DdaThreadedData::get()->registerRank(comm_->commHash, comm_->rank);

  // enable peer access (support for NVS full-mesh topology only)
  for (int i = 0; i < comm_->nRanks; ++i) {
    if (i == comm_->rank) {
      continue;
    }
    cudaError_t e = cudaDeviceEnablePeerAccess(i, 0);
    if (e != cudaErrorPeerAccessAlreadyEnabled && e != cudaSuccess) {
      CUDACHECKIGNORE(e);
    }
  }

  // allocate device memory
  const size_t kNumBarriers = 3;
  CUDACHECKIGNORE(cudaMalloc(
      &barrierMbox_d_, kNumBarriers * comm_->nRanks * sizeof(uintptr_t)));
  CUDACHECKIGNORE(cudaMemset(
      barrierMbox_d_, 0, kNumBarriers * comm_->nRanks * sizeof(uintptr_t)));
  CUDACHECKIGNORE(cudaMalloc(&tmpbuff_d_, 128 * 1024));
  CUDACHECKIGNORE(
      cudaMalloc(&devStates_d_, sizeof(DdaDeviceState) * comm_->nRanks));

  memHandler_.add(barrierMbox_d_);
  memHandler_.add(tmpbuff_d_);
  memHandler_.exchangeMemHandles();

  DdaDeviceState devStates[comm_->nRanks];
  for (int rank = 0; rank < comm_->nRanks; ++rank) {
    devStates[rank].barrierMbox =
        static_cast<uintptr_t*>(memHandler_.get(rank, 0));
    devStates[rank].tmpbuff = memHandler_.get(rank, 1);
  }

  CUDACHECKIGNORE(cudaMemcpy(
      devStates_d_,
      devStates,
      sizeof(DdaDeviceState) * comm_->nRanks,
      cudaMemcpyDefault));

  INFO(NCCL_COLL, "AlgoManager initialized.");
}

AlgoManager::~AlgoManager() {
  // unregister rank
  DdaThreadedData::get()->unregisterRank(comm_->commHash, comm_->rank);

  CUDACHECKIGNORE(cudaFree(barrierMbox_d_));
  CUDACHECKIGNORE(cudaFree(tmpbuff_d_));
  CUDACHECKIGNORE(cudaFree(devStates_d_));
}

std::unique_ptr<AllReduceAlgo> AlgoManager::getAllReduceAlgo(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  // add algorithm selection logic here
  return getAllReduceDdaNvsFlatThreadedAlgo(
      sendbuff, recvbuff, count, datatype, op, comm, stream);
}

std::unique_ptr<AllReduceDdaNvsFlatThreadedAlgo>
AlgoManager::getAllReduceDdaNvsFlatThreadedAlgo(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  // toggle barrier flag
  barrierFlag_ = !barrierFlag_;
  auto algo = std::unique_ptr<AllReduceDdaNvsFlatThreadedAlgo>(
      new AllReduceDdaNvsFlatThreadedAlgo{
          sendbuff,
          recvbuff,
          count,
          datatype,
          op,
          comm,
          stream,
          devStates_d_,
          barrierFlag_});
  return algo;
}

} // namespace algorithms
} // namespace nccl
