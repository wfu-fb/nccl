// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <CtranGpe.h>
#include <nccl.h>
#include <cstddef>
#include "AllToAllvImpl.h"
#include "Ctran.h"
#include "comm.h"

/*
=== BEGIN_NCCL_CVAR_INFO_BLOCK ===

 - name        : NCCL_CTRAN_ALLTOALLV_NUM_THREAD_BLOCKS
   type        : int
   default     : 64
   description : |-
     Number of thread blocks used for intra-node AllToAllv kernel.
     Must be even number.

 - name        : NCCL_CTRAN_ALLTOALLV_THREAD_BLOCK_SIZE
   type        : int
   default     : 640
   description : |-
     Number of threads in each thread block used for intra-node
     AllToAllv kernel.

=== END_NCCL_CVAR_INFO_BLOCK ===
*/

static void* alltoallvKerns[ncclNumTypes] = {
    (void*)ncclKernelAllToAllv<int8_t>,
    (void*)ncclKernelAllToAllv<uint8_t>,
    (void*)ncclKernelAllToAllv<int32_t>,
    (void*)ncclKernelAllToAllv<uint32_t>,
    (void*)ncclKernelAllToAllv<int64_t>,
    (void*)ncclKernelAllToAllv<uint64_t>,
    (void*)ncclKernelAllToAllv<half>,
    (void*)ncclKernelAllToAllv<float>,
    (void*)ncclKernelAllToAllv<double>,
#if defined(__CUDA_BF16_TYPES_EXIST__) && \
    NCCL_VERSION_CODE >= NCCL_VERSION(2, 10, 0)
    (void*)ncclKernelAllToAllv<__nv_bfloat16>,
#endif
#if defined(__CUDA_FP8_TYPES_EXIST__) && defined(NCCL_ENABLE_FP8)
    (void*)ncclKernelAllToAllv<__nv_fp8_e4m3>,
    (void*)ncclKernelAllToAllv<__nv_fp8_e5m2>,
#endif
};

static inline ncclResult_t setupKernelConfig(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream,
    KernelConfig& config) {
  // Unlike alltoall, we cannot automatically detect grid size because each rank
  // may see different counts; use static gridSize for now.
  config.numThreads = NCCL_CTRAN_ALLTOALLV_THREAD_BLOCK_SIZE;
  config.numBlocks = NCCL_CTRAN_ALLTOALLV_NUM_THREAD_BLOCKS;

  // Adjust gridSize to fit alltoallv kernel algorithm:
  // 1. gridSize must be even number, because we split blocks into two sets of
  //   groups, one for sends and the other for receives, each send and receive
  //   pair must use the same number of blocks
  if (config.numBlocks % 2) {
    config.numBlocks += 1;
  }
  // 2. gridSize must be <= CTRAN_ALGO_MAX_THREAD_BLOCKS, since internal
  //   states/flags holds at most CTRAN_ALGO_MAX_THREAD_BLOCKS blocks
  if (config.numBlocks < 2 || config.numBlocks > CTRAN_ALGO_MAX_THREAD_BLOCKS) {
    config.numBlocks = CTRAN_ALGO_MAX_THREAD_BLOCKS;
  }

  config.args.devState_d = comm->ctran->algo->devState_d;
  config.args.collective.alltoallv.sendbuff = sendbuff;
  config.args.collective.alltoallv.recvbuff = recvbuff;
  config.args.collective.alltoallv.selfCount = sendcounts[comm->rank];
  config.args.collective.alltoallv.selfSendDispl = sdispls[comm->rank];
  config.args.collective.alltoallv.selfRecvDispl = rdispls[comm->rank];

  // Pass number of thread block groups to kernel p2p elements
  // - Half blocks handle send, and the other handle receive
  // - Used in p2p elem to ensure ngroups number of inuse flags are checked when
  // reclaiming. This avoids cross-block sync in kernel
  const int ngroups = config.numBlocks / 2;
  comm->ctran->gpe->allocKernelP2pElems(
      comm->localRanks - 1,
      ngroups,
      &config.args.collective.alltoallv.sendElemsList);
  comm->ctran->gpe->allocKernelP2pElems(
      comm->localRanks - 1,
      ngroups,
      &config.args.collective.alltoallv.recvElemsList);

  // Ensure each rank sends to different peer at a time to avoid alltoone P2P
  // write congestion. For example, with localRanks = 4, the following
  // schedule is used:
  // - Round0:
  // rank0: s(1)r(3); rank1: s(2)r(0); rank2: s(3)r(1); rank3: s(0)r(2)
  // - Round1:
  // rank0: s(2)r(2); rank1: s(3)r(3); rank2: s(0)r(0); rank3: s(1)r(1)
  // - Round2:
  // rank0: s(3)r(1); rank1: s(0)r(2); rank2: s(1)r(3); rank3: s(2)r(0)
  KernelP2pElem* sendElem = config.args.collective.alltoallv.sendElemsList;
  KernelP2pElem* recvElem = config.args.collective.alltoallv.recvElemsList;
  for (int r = 0; r < comm->localRanks - 1; r++) {
    int sendPeer = (comm->localRank + r + 1) % comm->localRanks;
    int recvPeer =
        (comm->localRank + comm->localRanks - r - 1) % comm->localRanks;
    int sendPeerGlobal = comm->localRankToRank[sendPeer];
    int recvPeerGlobal = comm->localRankToRank[recvPeer];

    sendElem->peerRank = sendPeer;
    sendElem->count = sendcounts[sendPeerGlobal];
    sendElem->displ = sdispls[sendPeerGlobal];
    sendElem = sendElem->next;

    recvElem->peerRank = recvPeer;
    recvElem->count = recvcounts[recvPeerGlobal];
    recvElem->displ = rdispls[recvPeerGlobal];
    recvElem = recvElem->next;
  }

  return ncclSuccess;
}

static ncclResult_t opIbImpl(
    std::vector<std::unique_ptr<struct OpElem>> opGroup) {
  struct OpElem* op = opGroup.front().get();
  ncclComm_t comm = opGroup.front()->comm;

  std::unique_ptr<CtranMapperTimestamp> timestamp =
      std::unique_ptr<CtranMapperTimestamp>(
          new CtranMapperTimestamp("CtranAlltoAll"));

  return ctranAllToAllvIbImpl(
      op->alltoallv.sendbuff,
      op->alltoallv.sendcounts,
      op->alltoallv.sdispls,
      op->alltoallv.recvbuff,
      op->alltoallv.recvcounts,
      op->alltoallv.rdispls,
      op->alltoallv.datatype,
      comm,
      std::move(timestamp));
}

static inline ncclResult_t setupGpeOp(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    ncclDataType_t datatype,
    ncclComm_t comm,
    std::vector<std::unique_ptr<struct OpElem>>& opGroup) {
  // Passing op only when remote peers are present
  if (comm->localRanks < comm->nRanks) {
    std::unique_ptr<struct OpElem> op = std::unique_ptr<struct OpElem>(
        new OpElem(OpElem::opType::ALLTOALLV, comm));
    op->alltoallv.sendbuff = sendbuff;
    op->alltoallv.recvbuff = recvbuff;
    op->alltoallv.datatype = datatype;
    op->alltoallv.sendcounts.resize(comm->nRanks, 0);
    op->alltoallv.sdispls.resize(comm->nRanks, 0);
    op->alltoallv.recvcounts.resize(comm->nRanks, 0);
    op->alltoallv.rdispls.resize(comm->nRanks, 0);

    size_t totalSendCount = 0, totalRecvCount = 0;
    int myNode = comm->rankToNode[comm->rank];
    for (int i = 0; i < comm->nRanks; i++) {
      int peerNode = comm->rankToNode[i];
      // GPE thread handles only remote peers
      if (myNode != peerNode) {
        op->alltoallv.sendcounts[i] = sendcounts[i];
        op->alltoallv.sdispls[i] = sdispls[i];
        op->alltoallv.recvcounts[i] = recvcounts[i];
        op->alltoallv.rdispls[i] = rdispls[i];

        totalSendCount += sendcounts[i];
        totalRecvCount += recvcounts[i];
      }
    }
    // if contains either non-zero send or receive, pass op
    if (totalSendCount || totalRecvCount) {
      opGroup.push_back(std::move(op));
    }
  }
  return ncclSuccess;
}

ncclResult_t ctranAllToAllv(
    const void* sendbuff,
    const size_t sendcounts[],
    const size_t sdispls[],
    void* recvbuff,
    const size_t recvcounts[],
    const size_t rdispls[],
    ncclDataType_t datatype,
    ncclComm_t comm,
    cudaStream_t stream) {
  CTRAN_COLL_INFO(
      "ctranAllToAllv", sendbuff, recvbuff, 0UL, datatype, -1, comm, stream);
  for (int i = 0; i < comm->nRanks; i++) {
    INFO(
        NCCL_COLL,
        "%s: opCount %lx - sendcounts[%d] %ld sdispls[%d] %ld recvcounts[%d] %ld rdispls[%d] %ld",
        "ctranAllToAllv",
        comm->opCount - 1,
        i,
        sendcounts[i],
        i,
        sdispls[i],
        i,
        recvcounts[i],
        i,
        rdispls[i]);
  }

  // prepare kernel config for self and NVL copies
  KernelConfig config =
      KernelConfig(KernelConfig::KernelType::ALLTOALLV, stream);
  NCCLCHECK(setupKernelConfig(
      sendbuff,
      sendcounts,
      sdispls,
      recvbuff,
      recvcounts,
      rdispls,
      datatype,
      comm,
      stream,
      config));

  // prepare operation for IB path
  std::vector<std::unique_ptr<struct OpElem>> opGroup;
  NCCLCHECK(setupGpeOp(
      sendbuff,
      sendcounts,
      sdispls,
      recvbuff,
      recvcounts,
      rdispls,
      datatype,
      comm,
      opGroup));

  NCCLCHECK(comm->ctran->gpe->submit(
      std::move(opGroup),
      opIbImpl,
      config,
      reinterpret_cast<void*>(alltoallvKerns[datatype])));

  return ncclSuccess;
}
