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

 - name        : NCCL_DDA2_ALLREDUCE_MAX_BLOCKS
   type        : int
   default     : 24
   description : |-
     DDA Allreduce max number of blocks.

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
  maxBlocks_ =
      std::min(NCCL_DDA2_ALLREDUCE_MAX_BLOCKS, devProp_.multiProcessorCount);

  // allocate host memory
  devStates_ = static_cast<DdaDeviceState*>(
      malloc(sizeof(DdaDeviceState) * comm_->nRanks));

  // allocate device memory
  // we need 3 barriers for Tree Algo
  // 1) [r0, r1, ..., rN-1]
  // 2) [r0, r1, ..., rN-1]@block0, [r0, r1, ..., rN-1]@block1, ...@blockK-1
  // 3) [r0, r1, ..., rN-1]@block0, [r0, r1, ..., rN-1]@block1, ...@blockK-1
  CUDACHECKIGNORE(cudaMalloc(
      &barrierMbox_d_,
      (comm_->nRanks + 2 * maxBlocks_ * comm_->nRanks) * sizeof(uintptr_t)));
  CUDACHECKIGNORE(cudaMemset(
      barrierMbox_d_,
      0,
      (comm_->nRanks + 2 * maxBlocks_ * comm_->nRanks) * sizeof(uintptr_t)));

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

bool AlgoManager::checkNumRanks(size_t numRanks) {
  if (numRanks & (numRanks - 1)) {
    // power of two ranks
    return false;
  }

  if (numRanks == 1) {
    // more than 1 rank
    return false;
  }

  if (numRanks > 8) {
    // only support 2, 4, 8 ranks (single-node)
    return false;
  }

  return true;
}

bool AlgoManager::canRunDdaAllReduceThreaded(
    ncclComm* comm,
    ncclRedOp_t op,
    const void* sendbuff,
    void* recvbuff,
    size_t totalBytes,
    size_t numDdaThreads,
    size_t treeThresholdBytes) {
  if (numDdaThreads != comm->nRanks) {
    // my communicator group must contain only DdaThreads
    return false;
  }

  if (!checkNumRanks(comm->nRanks)) {
    return false;
  }

  if (op != ncclSum) {
    // only sum is supported
    return false;
  }

  if (((uintptr_t)sendbuff % 16) || ((uintptr_t)recvbuff % 16)) {
    // 16 byte alignment as we do 16-byte loads in DDA kernel
    return false;
  }

  if (totalBytes < treeThresholdBytes) {
    // Flat algo
    if (sendbuff == recvbuff) {
      // we don't support inplace FLAT threaded algo yet
      return false;
    }
    if (totalBytes % 16) {
      // 16-bytes load
      return false;
    }
  } else {
    // Tree algo
    if (totalBytes % (16 * comm->nRanks)) {
      // 16-bytes load
      return false;
    }
  }

  return true;
}

bool AlgoManager::canRunDdaAllReduceIpc(
    ncclComm* comm,
    ncclRedOp_t op,
    const void* sendbuff,
    void* recvbuff,
    size_t totalBytes,
    size_t treeThresholdBytes,
    size_t tmpbuffSize) {
  if (comm->localRanks != comm->nRanks) {
    // all ranks must be local
    return false;
  }

  if (!checkNumRanks(comm->nRanks)) {
    return false;
  }

  if (op != ncclSum) {
    // only sum is supported
    return false;
  }

  if (totalBytes < treeThresholdBytes) {
    // Flat algo
    if (totalBytes % 16) {
      // 16-bytes load
      return false;
    }
  } else {
    // Tree algo
    if (totalBytes % (16 * comm->nRanks)) {
      // 16-bytes load
      return false;
    }
  }

  if (totalBytes > tmpbuffSize) {
    // we always copy sendbuff to tmpbuff for IPC
    return false;
  }

  return true;
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
  const size_t numDdaThreads = DdaThreadedData::get()->numRanks(comm_->commHash);
  const size_t totalSize = count * getDataSize(datatype);

  if (numDdaThreads == comm_->nRanks) {
    // multi-threaded environment
    if (!canRunDdaAllReduceThreaded(
        comm,
        op,
        sendbuff,
        recvbuff,
        totalSize,
        numDdaThreads,
        NCCL_DDA2_ALLREDUCE_TREE_THRESHOLD_NVS)) {
      // fallback to default
      return nullptr;
    }

    if (totalSize < NCCL_DDA2_ALLREDUCE_TREE_THRESHOLD_NVS) {
      return getAllReduceDdaNvsFlatThreadedAlgo(
          sendbuff, recvbuff, count, datatype, op, comm, stream);
    } else {
      return getAllReduceDdaNvsTreeThreadedAlgo(
          sendbuff, recvbuff, count, datatype, op, comm, stream);
    }
  } else {
    // multi-process environment
    if (!canRunDdaAllReduceIpc(
          comm,
          op,
          sendbuff,
          recvbuff,
          totalSize,
          NCCL_DDA2_ALLREDUCE_TREE_THRESHOLD_NVS,
          NCCL_DDA2_ALLREDUCE_TMPBUFF_SIZE)) {
      // fallback to default
      return nullptr;
    }

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
          maxBlocks_});
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
          maxBlocks_});
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
          maxBlocks_});
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
          maxBlocks_});
  return algo;
}

} // namespace algorithms
} // namespace nccl
