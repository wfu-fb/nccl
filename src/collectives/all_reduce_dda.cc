// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <assert.h>
#include <cmath>
#include "argcheck.h"
#include "enqueue.h"
#include "nccl.h"

NCCL_PARAM(DDAAllreduceMaxBlocks, "DDA_ALLREDUCE_MAX_BLOCKS", 1);
NCCL_PARAM(DDAAllreduceTreeThresholdNVS, "DDA_ALLREDUCE_TREE_THRESHOLD_NVS", 256 * 1024);
NCCL_PARAM(DDAAllreduceTreeThresholdHCM, "DDA_ALLREDUCE_TREE_THRESHOLD_HCM", 256 * 1024);
NCCL_PARAM(DDAAllreduceLargeMessageHCM, "DDA_ALLREDUCE_LARGE_MESSAGE_HCM", 0);

#define ASSIGN_FUNC(func, templ, nranks)   \
  do {                                     \
    switch ((nranks)) {                    \
      case 2:                              \
        func = (const void*)&templ<T, 2>;  \
        break;                             \
                                           \
      case 4:                              \
        func = (const void*)&templ<T, 4>;  \
        break;                             \
                                           \
      case 8:                              \
        func = (const void*)&templ<T, 8>;  \
        break;                             \
                                           \
      case 16:                             \
        func = (const void*)&templ<T, 16>; \
        break;                             \
                                           \
      default:                             \
        return ncclInvalidUsage;           \
    }                                      \
  } while (0)

static inline int getMaxBlocks(ncclComm* comm) {
  int maxBlocks = ncclParamDDAAllreduceMaxBlocks();

  if (maxBlocks > comm->dda.devProp.multiProcessorCount) {
    WARN("NCCL_DDA_ALLREDUCE_MAX_BLOCKS cannot be larger than %d\n",
         comm->dda.devProp.multiProcessorCount);
    maxBlocks = comm->dda.devProp.multiProcessorCount;
  }

  return maxBlocks;
}

template <typename T>
static ncclResult_t launchKernel(
    ncclComm* comm,
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    cudaStream_t stream) {
  bool enableIpc = comm->dda.md->enableIpc();
  if (enableIpc) {
    CUDACHECK(cudaMemcpyAsync(
        comm->dda.md->tmpSendbuff,
        sendbuff,
        count * sizeof(T),
        cudaMemcpyDefault,
        stream));
  }
  const void* func;

  if (comm->dda.md->topoType == NCCL_DDA_TOPO_TYPE__NVS) {
    if (count * sizeof(T) < ncclParamDDAAllreduceTreeThresholdNVS()) {
      if (enableIpc) {
        ASSIGN_FUNC(func, ncclKernel_AllReduce_DDA_Flat_ipc, comm->nRanks);
      } else {
        ASSIGN_FUNC(func, ncclKernel_AllReduce_DDA_Flat, comm->nRanks);
      }
    } else {
      if (enableIpc) {
        ASSIGN_FUNC(func, ncclKernel_AllReduce_DDA_Tree_ipc, comm->nRanks);
      } else {
        ASSIGN_FUNC(func, ncclKernel_AllReduce_DDA_Tree, comm->nRanks);
      }
    }
  } else if (comm->dda.md->topoType == NCCL_DDA_TOPO_TYPE__HCM) {
    if (count * sizeof(T) < ncclParamDDAAllreduceTreeThresholdHCM()) {
      ASSIGN_FUNC(func, ncclKernel_AllReduce_DDA_HCM_Flat, comm->nRanks);
    } else if (ncclParamDDAAllreduceLargeMessageHCM()) {
      ASSIGN_FUNC(func, ncclKernel_AllReduce_DDA_HCM_Tree, comm->nRanks);
    } else {
      return ncclInvalidUsage;
    }
  } else {
    return ncclInvalidUsage;
  }

  cudaFuncAttributes attr;
  CUDACHECK(cudaFuncGetAttributes(&attr, func));

  int maxBlocks = getMaxBlocks(comm);
  int numBlocks[2] = {1, maxBlocks};
  int numThreads[2] = {32, attr.maxThreadsPerBlock};
  int eltsPerThread = 16 / sizeof(T);
  dim3 grid;
  dim3 blocks;

  if (count <= numBlocks[0] * numThreads[0] * eltsPerThread) {
    /* for small counts, use the minimum number of blocks and
     * threads, while keeping eltsPerThread elements to be
     * computed by each thread. */
    grid.x = numBlocks[0];
    blocks.x = numThreads[0];
  } else if (count <= numBlocks[0] * numThreads[1] * eltsPerThread) {
    /* for slightly larger counts, increase the number of threads
     * per block to up to the maximum number of threads. */
    grid.x = numBlocks[0];
    blocks.x = (int)std::ceil(count / (numBlocks[0] * eltsPerThread));
  } else if (count <= numBlocks[1] * numThreads[1] * eltsPerThread) {
    /* for even larger counts, increase the number of blocks to up
     * to the maximum number of blocks. */
    grid.x = (int)std::ceil(count / (numThreads[1] * eltsPerThread));
    blocks.x = numThreads[1];
  } else {
    /* for even larger counts, use the maximum number of threads
     * and blocks, and let each thread compute more elements. */
    grid.x = numBlocks[1];
    blocks.x = numThreads[1];
  }
  grid.y = 1;
  grid.z = 1;
  blocks.y = 1;
  blocks.z = 1;

  /* mbox_id,barrierFlag: 1,0; 0,1; 1,1; 0,0 */
  /* We maintain two mboxes (0,1) and two flags (0,1) for each mbox.
   * We start with an mboxId of 1, and a barrierFlag of 0.  And each
   * mbox is set to all zeroes.
   *
   * At this point, the last bit in each element of the mbox is 0.
   *
   * The first time we call this collective, we switch the mboxId to
   * 0, and the barrierFlag to 1.  This way, the barrier operation
   * in our kernel has to wait for the barrierFlag to move to 1
   * before it can exit the barrier.
   *
   * The second time we call this collective, we switch the mboxId
   * to 1, and the barrierFlag to 1.  This way, the barrier
   * operation in our kernel has to wait for the barrierFlag to move
   * to 1 before it can exit the barrier.
   *
   * At this point, the last bit in each element of the mbox is 1.
   *
   * The third time we call this collective, we switch the mboxId to
   * 0, and the barrierFlag to 0.  This way, the barrier operation
   * in our kernel has to wait for the barrierFlag to move to 0
   * before it can exit the barrier.
   *
   * The fourth time we call this collective, we switch the mboxId
   * to 1, and the barrierFlag to 0.  This way, the barrier
   * operation in our kernel has to wait for the barrierFlag to move
   * to 0 before it can exit the barrier.
   *
   * At this point, the last bit in each element of the mbox is 0,
   * and we are back to our original state.
   *
   * We need two mboxes, so one rank can get started with the next
   * collective even if not all ranks have exited the previous
   * collective (and thus are still using the previous mbox).
   */
  if (comm->dda.barrierMboxId == 1 && comm->dda.barrierFlag == 0) {
    comm->dda.barrierMboxId = !comm->dda.barrierMboxId;
    comm->dda.barrierFlag = !comm->dda.barrierFlag;
  } else if (comm->dda.barrierMboxId == 0 && comm->dda.barrierFlag == 1) {
    comm->dda.barrierMboxId = !comm->dda.barrierMboxId;
  } else if (comm->dda.barrierMboxId == 1 && comm->dda.barrierFlag == 1) {
    comm->dda.barrierMboxId = !comm->dda.barrierMboxId;
    comm->dda.barrierFlag = !comm->dda.barrierFlag;
  } else if (comm->dda.barrierMboxId == 0 && comm->dda.barrierFlag == 0) {
    comm->dda.barrierMboxId = !comm->dda.barrierMboxId;
  }

  if (comm->dda.md->topoType == NCCL_DDA_TOPO_TYPE__NVS) {
    ddaClique* clique = comm->dda.md->cliques.front();

    if (count * sizeof(T) < ncclParamDDAAllreduceTreeThresholdNVS()) {
      if (enableIpc) {
        void* args[] = {
            &comm->dda.md->barrierMbox[comm->dda.barrierMboxId],
            &comm->dda.barrierFlag,
            &comm->rank,
            &recvbuff,
            &count,
            &comm->dda.md->allTmpSendbuffs};
        CUDACHECK(cudaLaunchKernel(func, grid, blocks, args, 0, stream));
      } else {
        void* args[] = {
            &clique->barrierMbox[comm->dda.barrierMboxId],
            &comm->dda.barrierFlag,
            &comm->rank,
            &sendbuff,
            &recvbuff,
            &count};
        CUDACHECK(cudaLaunchKernel(func, grid, blocks, args, 0, stream));
      }
    } else {
      // scatter-reduce, allgather tree algorithm
      if (enableIpc) {
        void* args[] = {
            &comm->dda.md->barrierMbox[comm->dda.barrierMboxId],
            &comm->dda.barrierFlag,
            &comm->rank,
            &comm->dda.md->allTmpSendbuffs,
            &recvbuff,
            &count};
        CUDACHECK(cudaLaunchKernel(func, grid, blocks, args, 0, stream));
      } else {
        void* args[] = {
            &clique->barrierMbox[comm->dda.barrierMboxId],
            &comm->dda.barrierFlag,
            &comm->rank,
            &sendbuff,
            &recvbuff,
            &count};
        CUDACHECK(cudaLaunchKernel(func, grid, blocks, args, 0, stream));
      }
    }
  } else if (comm->dda.md->topoType == NCCL_DDA_TOPO_TYPE__HCM) {
    ddaClique* clique = comm->dda.md->cliques.front();
    ddaClique* peerClique = comm->dda.md->cliques.back();

    if (clique->rankToGpu.find(comm->rank) == clique->rankToGpu.end()) {
      clique = comm->dda.md->cliques.back();
      peerClique = comm->dda.md->cliques.front();
    }
    assert(clique->rankToGpu.find(comm->rank) != clique->rankToGpu.end());

    int cliqueRank;
    for (cliqueRank = 0; cliqueRank < clique->gpus.size(); cliqueRank++) {
      if (clique->rankToGpu[comm->rank] == clique->gpus[cliqueRank]) {
        break;
      }
    }
    assert(cliqueRank < clique->gpus.size());

    int peerRank = -1;
    for (auto it : peerClique->rankToGpu) {
      if (it.second == peerClique->gpus[cliqueRank]) {
        peerRank = it.first;
      }
    }
    assert(peerRank != -1);

    comm->dda.localMboxId = !comm->dda.localMboxId;

    assert(peerClique->rankToLocalMbox[comm->dda.localMboxId][peerRank] != nullptr);

    void* args[] = {
        &clique->barrierMbox[comm->dda.barrierMboxId],
        &clique->rankToLocalMbox[comm->dda.localMboxId][comm->rank],
        &peerClique->rankToLocalMbox[comm->dda.localMboxId][peerRank],
        &comm->dda.barrierFlag,
        &cliqueRank,
        &sendbuff,
        &clique->rankToTmpbuf[comm->rank],
        &recvbuff,
        &count};

    CUDACHECK(cudaLaunchKernel(func, grid, blocks, args, 0, stream));
  }

  return ncclSuccess;
}

ncclResult_t ncclAllReduceDDA(
    const void* sendbuff,
    void* recvbuff,
    size_t count,
    ncclDataType_t datatype,
    ncclRedOp_t op,
    ncclComm* comm,
    cudaStream_t stream) {
  ddaClique* clique;
  int numDDAThreads = 0;
  ncclResult_t res;

  NCCLCHECK(ncclCommEnsureReady(comm));
  if (datatype < 0 || datatype >= ncclNumTypes) {
    WARN("AllReduce : invalid type %d", datatype);
    return ncclInvalidArgument;
  }
  if (op < 0 || ncclMaxRedOp < op) {
    WARN("AllReduce : invalid reduction operation %d", op);
    return ncclInvalidArgument;
  }
  NCCLCHECK(CudaPtrCheck(sendbuff, comm, "sendbuff", "AllReduce"));
  NCCLCHECK(CudaPtrCheck(recvbuff, comm, "recvbuff", "AllReduce"));

  for (auto c : comm->dda.md->cliques) {
    numDDAThreads += c->rankToGpu.size();
  }

  const auto enableIpc = comm->dda.md->enableIpc();
  const auto bytes = count * typeSize(datatype);
  if (!enableIpc) {
    if ((numDDAThreads != comm->nRanks) || /* collective must only contain dda ranks */
        (numDDAThreads & (numDDAThreads - 1)) || /* power of two ranks */
        (numDDAThreads == 1) || /* more than one rank */
        (numDDAThreads > ncclParamMaxDDAThreads()) || /* only small rank counts are supported */
        (op != ncclSum) || /* only sum is supported */
        ((uintptr_t)sendbuff % 16) || /* 16-byte alignment */
        ((uintptr_t)recvbuff % 16)) { /* 16-byte alignment */
      goto not_supported;
    }

    if (comm->dda.md->topoType == NCCL_DDA_TOPO_TYPE__NVS) {
      if (bytes < ncclParamDDAAllreduceTreeThresholdNVS()) {
        if ((bytes % 16) || /* allow for 16-byte loads */
            (sendbuff == recvbuff)) { /* in-place reduction */
          goto not_supported;
        }
      } else { /* bytes >= ncclParamDDAAllreduceTreeThresholdNVS() */
        if (bytes % (16 * comm->nRanks)) { /* allow for 16-byte loads */
          goto not_supported;
        }
      }
    } else { /* comm->dda.md->topoType == NCCL_DDA_TOPO_TYPE__HCM */
      if (bytes > ncclParamDDAAllreduceTmpbuffSize()) { /* need tmpbuff */
        goto not_supported;
      }
    }
  } else { /* enableIpc */
    // TODO: check all processes belong to a single node
    if (comm->dda.md->topoType == NCCL_DDA_TOPO_TYPE__NVS) {
      if (bytes > ncclParamDDAAllreduceLocalBufSize()) { /* need tmpbuff for IPC */
        goto not_supported;
      }
    } else { /* comm->dda.md->topoType == NCCL_DDA_TOPO_TYPE__HCM */
      goto not_supported;
    }
  }

  switch (datatype) {
    case ncclInt8:
      NCCLCHECK(launchKernel<char>(comm, sendbuff, recvbuff, count, stream));
      break;

    case ncclUint8:
      NCCLCHECK(launchKernel<uint8_t>(comm, sendbuff, recvbuff, count, stream));
      break;

    case ncclInt32:
      NCCLCHECK(launchKernel<int32_t>(comm, sendbuff, recvbuff, count, stream));
      break;

    case ncclUint32:
      NCCLCHECK(
          launchKernel<uint32_t>(comm, sendbuff, recvbuff, count, stream));
      break;

    case ncclInt64:
      NCCLCHECK(launchKernel<int64_t>(comm, sendbuff, recvbuff, count, stream));
      break;

    case ncclUint64:
      NCCLCHECK(
          launchKernel<uint64_t>(comm, sendbuff, recvbuff, count, stream));
      break;

    case ncclFloat16:
      NCCLCHECK(launchKernel<half>(comm, sendbuff, recvbuff, count, stream));
      break;

    case ncclFloat32:
      NCCLCHECK(launchKernel<float>(comm, sendbuff, recvbuff, count, stream));
      break;

    case ncclFloat64:
      NCCLCHECK(launchKernel<double>(comm, sendbuff, recvbuff, count, stream));
      break;

#if defined(__CUDA_BF16_TYPES_EXIST__)
    case ncclBfloat16:
      NCCLCHECK(launchKernel<__nv_bfloat16>(comm, sendbuff, recvbuff, count, stream));
      break;
#endif

    default:
      goto not_supported;
  }

  return ncclSuccess;

not_supported:
  return ncclInvalidUsage;
}
