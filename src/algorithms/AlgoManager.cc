// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#include "AlgoManager.h"

#include "AlgoUtils.h"
#include "DdaThreadedData.h"
#include "argcheck.h"
#include "checks.h"
#include "comm.h"
#include "debug.h"
#include "nccl_cvars.h"

#include <cassert>

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_DDA2_ALLREDUCE_TREE_THRESHOLD_NVS
   type        : int
   default     : 262144
   description : |-
     Message size at which DDA Allreduce switches to the tree algorithm.
     Only applies for NVSwitch-based systems.

 - name        : NCCL_DDA2_ALLREDUCE_TMPBUFF_SIZE
   type        : int
   default     : 33554432
   description : |-
     DDA Allreduce temporary buffer size.

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

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

  // get device property (expensive call: 10+ ms)
  CUDACHECKIGNORE(cudaGetDeviceProperties(&devProp_, comm_->cudaDev));
  const int kMaxBlocks = devProp_.multiProcessorCount;

  // allocate host memory
  devStates_ = static_cast<DdaDeviceState*>(
      malloc(sizeof(DdaDeviceState) * comm_->nRanks));

  // each barrier has following memory layout
  // rank0:[b0, b1, ..., bk], rank1:[b0, b1, ..., bk]
  // where bk represents k-th block

  // allocate device memory
  // we need 3 barriers for tree algorithms
  // barrier, RS, barrier, AG, barrier
  const size_t kNumBarriers = 3;
  CUDACHECKIGNORE(cudaMalloc(
      &barrierMbox_d_, kNumBarriers * comm_->nRanks * kMaxBlocks * sizeof(uintptr_t)));
  CUDACHECKIGNORE(cudaMemset(
      barrierMbox_d_, 0, kNumBarriers * comm_->nRanks * kMaxBlocks * sizeof(uintptr_t)));
  CUDACHECKIGNORE(cudaMalloc(&tmpbuff_d_, NCCL_DDA2_ALLREDUCE_TMPBUFF_SIZE));
  CUDACHECKIGNORE(
      cudaMalloc(&devStates_d_, sizeof(DdaDeviceState) * comm_->nRanks));

  // exchange handles
  memHandler_.add(barrierMbox_d_);
  memHandler_.add(tmpbuff_d_);
  memHandler_.exchangeMemHandles();
  for (int rank = 0; rank < comm_->nRanks; ++rank) {
    devStates_[rank].barrierMbox =
        static_cast<uintptr_t*>(memHandler_.get(rank, 0));
    devStates_[rank].tmpbuff = memHandler_.get(rank, 1);
  }

  CUDACHECKIGNORE(cudaMemcpy(
      devStates_d_,
      devStates_,
      sizeof(DdaDeviceState) * comm_->nRanks,
      cudaMemcpyDefault));

  INFO(NCCL_INIT, "AlgoManager initialized.");
}

AlgoManager::~AlgoManager() {
  // unregister rank
  DdaThreadedData::get()->unregisterRank(comm_->commHash, comm_->rank);

  // free device memory
  CUDACHECKIGNORE(cudaFree(barrierMbox_d_));
  CUDACHECKIGNORE(cudaFree(tmpbuff_d_));
  CUDACHECKIGNORE(cudaFree(devStates_d_));

  // free host memory
  free(devStates_);

  INFO(NCCL_INIT, "AlgoManager destroyed.");
}

std::unique_ptr<AllReduceAlgo> AlgoManager::getAllReduceAlgo(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  // select proper algorithm
  const size_t numThreadedRanks = DdaThreadedData::get()->numRanks(comm_->commHash);
  const size_t totalSize = count * getDataSize(datatype);

  if (numThreadedRanks == comm_->nRanks) {
    // multi-threaded environment
    if (totalSize < NCCL_DDA2_ALLREDUCE_TREE_THRESHOLD_NVS) {
      return getAllReduceDdaNvsFlatThreadedAlgo(
          sendbuff, recvbuff, count, datatype, op, comm, stream);
    } else {
      return getAllReduceDdaNvsTreeThreadedAlgo(
          sendbuff, recvbuff, count, datatype, op, comm, stream);
    }
  } else {
    // multi-process environment

    // copy src to tmp buffers
    assert(totalSize <= NCCL_DDA2_ALLREDUCE_TMPBUFF_SIZE);
    CUDACHECKIGNORE(cudaMemcpyAsync(
        devStates_[comm_->rank].tmpbuff,
        sendbuff,
        totalSize,
        cudaMemcpyDefault,
        stream));

    if (totalSize < NCCL_DDA2_ALLREDUCE_TREE_THRESHOLD_NVS) {
      return getAllReduceDdaNvsFlatIpcAlgo(
          sendbuff, recvbuff, count, datatype, op, comm, stream);
    } else {
      return getAllReduceDdaNvsTreeIpcAlgo(
          sendbuff, recvbuff, count, datatype, op, comm, stream);
    }
  }
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
          barrierFlag_,
          devProp_.multiProcessorCount});
  return algo;
}

std::unique_ptr<AllReduceDdaNvsTreeThreadedAlgo>
AlgoManager::getAllReduceDdaNvsTreeThreadedAlgo(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  // toggle barrier flag
  barrierFlag_ = !barrierFlag_;
  auto algo = std::unique_ptr<AllReduceDdaNvsTreeThreadedAlgo>(
      new AllReduceDdaNvsTreeThreadedAlgo{
          sendbuff,
          recvbuff,
          count,
          datatype,
          op,
          comm,
          stream,
          devStates_d_,
          barrierFlag_,
          devProp_.multiProcessorCount});
  return algo;
}

std::unique_ptr<AllReduceDdaNvsFlatIpcAlgo>
AlgoManager::getAllReduceDdaNvsFlatIpcAlgo(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  // toggle barrier flag
  barrierFlag_ = !barrierFlag_;
  auto algo = std::unique_ptr<AllReduceDdaNvsFlatIpcAlgo>(
      new AllReduceDdaNvsFlatIpcAlgo{
          sendbuff,
          recvbuff,
          count,
          datatype,
          op,
          comm,
          stream,
          devStates_d_,
          barrierFlag_,
          devProp_.multiProcessorCount});
  return algo;
}

std::unique_ptr<AllReduceDdaNvsTreeIpcAlgo>
AlgoManager::getAllReduceDdaNvsTreeIpcAlgo(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  // toggle barrier flag
  barrierFlag_ = !barrierFlag_;
  auto algo = std::unique_ptr<AllReduceDdaNvsTreeIpcAlgo>(
      new AllReduceDdaNvsTreeIpcAlgo{
          sendbuff,
          recvbuff,
          count,
          datatype,
          op,
          comm,
          stream,
          devStates_d_,
          barrierFlag_,
          devProp_.multiProcessorCount});
  return algo;
}

} // namespace algorithms
} // namespace nccl
