// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <CtranGpe.h>
#include <nccl.h>
#include <cstddef>
#include <memory>
#include <vector>
#include "AllToAllvImpl.h"
#include "Ctran.h"
#include "CtranAlgoDev.h"
#include "comm.h"

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_CTRAN_ALLTOALL_NUM_THREAD_BLOCKS
   type        : int
   default     : -1
   description : |-
     Number of thread blocks to use for AllToAll.
     Setting it to a negative number means that NCCL will automatically
     pick a value.

 - name        : NCCL_CTRAN_ALLTOALL_THREAD_BLOCK_SIZE
   type        : int
   default     : -1
   description : |-
     Number of threads in each thread block to use for AllToAll.
     Setting it to a negative number means that NCCL will automatically
     pick a value.

 - name        : NCCL_CTRAN_ALLTOALL_THRESHOLD
   type        : uint64_t
   default     : 32768
   description : |-
     Minimal message size in bytes to send to (receive from) each rank to use
     CTran AllToAll. Messages smaller than the threshold may benefit from
     the default eager copy based algorithm.

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

static void* alltoallKerns[ncclNumTypes] = {
    (void*)ncclKernelAllToAll<int8_t>,
    (void*)ncclKernelAllToAll<uint8_t>,
    (void*)ncclKernelAllToAll<int32_t>,
    (void*)ncclKernelAllToAll<uint32_t>,
    (void*)ncclKernelAllToAll<int64_t>,
    (void*)ncclKernelAllToAll<uint64_t>,
    (void*)ncclKernelAllToAll<half>,
    (void*)ncclKernelAllToAll<float>,
    (void*)ncclKernelAllToAll<double>,
#if defined(__CUDA_BF16_TYPES_EXIST__) && \
    NCCL_VERSION_CODE >= NCCL_VERSION(2, 10, 0)
    (void*)ncclKernelAllToAll<__nv_bfloat16>,
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
    (void*)ncclKernelAllToAll<__nv_fp8_e4m3>,
    (void*)ncclKernelAllToAll<__nv_fp8_e5m2>,
#endif
};

static ncclResult_t opIbImpl(
    std::vector<std::unique_ptr<struct OpElem>> opGroup) {
  struct OpElem* op = opGroup.front().get();
  ncclComm_t comm = opGroup.front()->comm;

  std::vector<size_t> sendcounts(comm->nRanks, 0);
  std::vector<size_t> sdispls(comm->nRanks, 0);
  std::vector<size_t> recvcounts(comm->nRanks, 0);
  std::vector<size_t> rdispls(comm->nRanks, 0);

  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp("CtranAlltoAll"));

  int myNode = comm->rankToNode[comm->rank];
  for (int i = 0; i < comm->nRanks; i++) {
    int peerNode = comm->rankToNode[i];
    // GPE thread handles only remote peers
    if (myNode != peerNode) {
      sendcounts[i] = op->alltoall.count;
      sdispls[i] = op->alltoall.count * i;
      recvcounts[i] = op->alltoall.count;
      rdispls[i] = op->alltoall.count * i;
    }
  }

  return ctranAllToAllvIbImpl(
      op->alltoall.sendbuff,
      sendcounts,
      sdispls,
      op->alltoall.recvbuff,
      recvcounts,
      rdispls,
      op->alltoall.datatype,
      comm,
      std::move(timestamp));
}

static unsigned int bestThreadBlockSize = 0;
static inline ncclResult_t setupKernelConfig(
    const void* sendbuff,
    void* recvbuff,
    const size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream,
    KernelConfig& config) {
  // If first time call, query cuda recommended blockSize
  if (bestThreadBlockSize == 0) {
    int minGridSize = 0;
    CUDACHECK(cudaOccupancyMaxPotentialBlockSize(
        &minGridSize,
        (int*)&bestThreadBlockSize,
        reinterpret_cast<const void*>(alltoallKerns[datatype])));
  }

  // Allow user to customize thread block size if specified
  config.numThreads = NCCL_CTRAN_ALLTOALL_THREAD_BLOCK_SIZE > 0
      ? NCCL_CTRAN_ALLTOALL_THREAD_BLOCK_SIZE
      : bestThreadBlockSize;

  // Use specified grid size if specified and in limit; otherwise use default
  if (NCCL_CTRAN_ALLTOALL_NUM_THREAD_BLOCKS < 1 ||
      NCCL_CTRAN_ALLTOALL_NUM_THREAD_BLOCKS > CTRAN_ALGO_MAX_THREAD_BLOCKS) {
    // Calculate default grid size based on block size
    unsigned int gridSize = (count + config.numThreads - 1) / config.numThreads;
    if (gridSize > CTRAN_ALGO_MAX_THREAD_BLOCKS) {
      gridSize = CTRAN_ALGO_MAX_THREAD_BLOCKS;
    }
    config.numBlocks = gridSize;
  } else {
    config.numBlocks = NCCL_CTRAN_ALLTOALL_NUM_THREAD_BLOCKS;
  }

  // gridSize must be even number, because we split blocks into two sets of
  // groups, one for sends and the other for receives, each send and receive
  // pair must use the same number of blocks
  if (config.numBlocks % 2) {
    config.numBlocks += 1;
  }

  config.args.devState_d = comm->ctran->algo->devState_d;
  config.args.collective.alltoall.sendbuff = sendbuff;
  config.args.collective.alltoall.recvbuff = recvbuff;
  config.args.collective.alltoall.count = count;

  return ncclSuccess;
}

static inline ncclResult_t setupGpeOp(
    const void* sendbuff,
    void* recvbuff,
    const size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream,
    std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  // Passing op only when remote peers are present
  if (comm->nNodes > 1) {
    std::unique_ptr<struct OpElem> op = std::unique_ptr<struct OpElem>(
        new OpElem(OpElem::opType::ALLTOALL, stream, comm));
    op->alltoall.sendbuff = sendbuff;
    op->alltoall.recvbuff = recvbuff;
    op->alltoall.count = count;
    op->alltoall.datatype = datatype;
    opGroup.push_back(std::move(op));
  }

  return ncclSuccess;
}

ncclResult_t ctranAllToAll(
    const void* sendbuff,
    void* recvbuff,
    const size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
  CTRAN_COLL_INFO(
      "CtranAllToAll", sendbuff, recvbuff, count, datatype, -1, comm, stream);

  if (count == 0) {
    return ncclSuccess;
  }

  // TODO: alltoallKerns perform poorly on HCM due to lack of NVL connection
  // between some GPUs We need detect topology and switch to use IB transport in
  // such a case

  // prepare kernel config for self and NVL copies
  KernelConfig config =
      KernelConfig(KernelConfig::KernelType::ALLTOALL, stream);
  NCCLCHECK(setupKernelConfig(
      sendbuff, recvbuff, count, datatype, comm, stream, config));

  // prepare operation for IB path
  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  NCCLCHECK(
      setupGpeOp(sendbuff, recvbuff, count, datatype, comm, stream, opGroup));

  NCCLCHECK(comm->ctran->gpe->submit(
      std::move(opGroup),
      opIbImpl,
      config,
      reinterpret_cast<void*>(alltoallKerns[datatype])));

  return ncclSuccess;
}

bool ctranAllToAllSupport(
    const size_t count,
    ncclDataType_t datatype,
    ncclComm_t comm) {
  if (ctranInitialized(comm) &&
      ncclTypeSize(datatype) * count >= NCCL_CTRAN_ALLTOALL_THRESHOLD) {
    return true;
  } else {
    return false;
  }
}
