// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
#include "AlgoManagerAllReduce.h"

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

 - name        : NCCL_DDA_ALLREDUCE_TREE_THRESHOLD
   type        : uint64_t
   default     : 262144
   description : |-
     Message size at which DDA Allreduce switches to the tree algorithm.

 - name        : NCCL_DDA_ALLREDUCE_SCATGAT_THRESHOLD
   type        : uint64_t
   default     : 1048576
   description : |-
     Message size at which DDA Allreduce switches to the scatter-gather algorithm.

 - name        : NCCL_DDA_ALLREDUCE_MAX_BLOCKS
   type        : int
   default     : 24
   description : |-
     DDA Allreduce max number of blocks.

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

namespace nccl {
namespace algorithms {

bool AlgoManagerAllReduce::canRunDdaAllReduceThreaded(
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

bool AlgoManagerAllReduce::canRunDdaAllReduceIpc(
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

std::unique_ptr<AlgoAllReduce> AlgoManagerAllReduce::getAlgoAllReduce(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  // select proper algorithm
  const size_t numDdaThreads = DdaThreadedData::get()->numRanks(comm_->commHash);
  const size_t totalSize = count * ncclTypeSize(datatype);

  if (numDdaThreads == comm_->nRanks) {
    // multi-threaded environment
    if (!canRunDdaAllReduceThreaded(
        comm,
        op,
        sendbuff,
        recvbuff,
        totalSize,
        numDdaThreads,
        NCCL_DDA_ALLREDUCE_TREE_THRESHOLD)) {
      // fallback to default
      return nullptr;
    }

    if (totalSize < NCCL_DDA_ALLREDUCE_TREE_THRESHOLD) {
      return getAlgoAllReduceDdaNvsFlatThreaded(
          sendbuff, recvbuff, count, datatype, op, comm, stream);
    } else {
      return getAlgoAllReduceDdaNvsTreeThreaded(
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
          NCCL_DDA_ALLREDUCE_TREE_THRESHOLD,
          NCCL_DDA_TMPBUFF_SIZE)) {
      // fallback to default
      return nullptr;
    }

    assert(totalSize <= NCCL_DDA_TMPBUFF_SIZE);
    if (totalSize < NCCL_DDA_ALLREDUCE_TREE_THRESHOLD) {
      return getAlgoAllReduceDdaNvsFlatIpc(
          sendbuff, recvbuff, count, datatype, op, comm, stream);
    } else if (totalSize < NCCL_DDA_ALLREDUCE_SCATGAT_THRESHOLD) {
      return getAlgoAllReduceDdaNvsTreeIpc(
          sendbuff, recvbuff, count, datatype, op, comm, stream);
    } else {
      return getAlgoAllReduceDdaNvsScatGatIpc(
          sendbuff, recvbuff, count, datatype, op, comm, stream);
    }
  }
}

std::unique_ptr<AlgoAllReduceDdaNvsFlatThreaded>
AlgoManagerAllReduce::getAlgoAllReduceDdaNvsFlatThreaded(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  // toggle barrier flag
  threadedBarrierFlag_ = !threadedBarrierFlag_;
  auto algo = std::unique_ptr<AlgoAllReduceDdaNvsFlatThreaded>(
      new AlgoAllReduceDdaNvsFlatThreaded{
          sendbuff,
          recvbuff,
          count,
          datatype,
          op,
          comm,
          stream,
          devStates_d_,
          threadedBarrierFlag_,
          maxBlocks_});
  return algo;
}

std::unique_ptr<AlgoAllReduceDdaNvsTreeThreaded>
AlgoManagerAllReduce::getAlgoAllReduceDdaNvsTreeThreaded(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  // toggle barrier flag
  threadedBarrierFlag_ = !threadedBarrierFlag_;
  auto algo = std::unique_ptr<AlgoAllReduceDdaNvsTreeThreaded>(
      new AlgoAllReduceDdaNvsTreeThreaded{
          sendbuff,
          recvbuff,
          count,
          datatype,
          op,
          comm,
          stream,
          devStates_d_,
          threadedBarrierFlag_,
          maxBlocks_});
  return algo;
}

std::unique_ptr<AlgoAllReduceDdaNvsFlatIpc>
AlgoManagerAllReduce::getAlgoAllReduceDdaNvsFlatIpc(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  auto algo = std::unique_ptr<AlgoAllReduceDdaNvsFlatIpc>(
      new AlgoAllReduceDdaNvsFlatIpc{
          sendbuff,
          recvbuff,
          count,
          datatype,
          op,
          comm,
          stream,
          devStates_,
          devStates_d_,
          ipcBarrierFlag_,
          maxBlocks_});

  // increment barrier flag (FlatIpc uses two barriers)
  ipcBarrierFlag_ += 2;

  return algo;
}

std::unique_ptr<AlgoAllReduceDdaNvsTreeIpc>
AlgoManagerAllReduce::getAlgoAllReduceDdaNvsTreeIpc(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  auto algo = std::unique_ptr<AlgoAllReduceDdaNvsTreeIpc>(
      new AlgoAllReduceDdaNvsTreeIpc{
          sendbuff,
          recvbuff,
          count,
          datatype,
          op,
          comm,
          stream,
          devStates_,
          devStates_d_,
          ipcBarrierFlag_,
          maxBlocks_});

  // increment barrier flag (TreeIpc uses three barriers)
  ipcBarrierFlag_ += 3;

  return algo;
}

std::unique_ptr<AlgoAllReduceDdaNvsScatGatIpc>
AlgoManagerAllReduce::getAlgoAllReduceDdaNvsScatGatIpc(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  auto algo = std::unique_ptr<AlgoAllReduceDdaNvsScatGatIpc>(
      new AlgoAllReduceDdaNvsScatGatIpc{
          sendbuff,
          recvbuff,
          count,
          datatype,
          op,
          comm,
          stream,
          devStates_,
          devStates_d_,
          ipcBarrierFlag_,
          maxBlocks_});

  // increment barrier flag (ScatGatIpc uses four barriers)
  ipcBarrierFlag_ += 4;

  return algo;
}

} // namespace algorithms
} // namespace nccl
