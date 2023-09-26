// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <assert.h>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>
#include "checks.h"
#include "comm.h"
#include "nccl.h"
#include "graph/topo.h"

NCCL_PARAM(DDAAllreduceTmpbuffSize, "DDA_ALLREDUCE_TMPBUFF_SIZE", 32 * 1024 * 1024);
NCCL_PARAM(MaxDDAThreads, "MAX_DDA_THREADS", 16);
NCCL_PARAM(ForceP2pAccess, "FORCE_P2P_ACCESS", 0);

static std::vector<ddaThreadSharedMd*> ddaThreadSharedMdList;
static std::mutex ddaThreadSharedMdListMutex;

bool operator==(const ncclUniqueId& lhs, const ncclUniqueId& rhs) {
  for (int i = 0; i < sizeof(ncclUniqueId); i++) {
    if (lhs.internal[i] != rhs.internal[i]) {
      return false;
    }
  }

  return true;
}

bool operator==(const ddaThreadSharedMd& lhs, const ddaThreadSharedMd& rhs) {
  return (lhs.commId == rhs.commId);
}

/* This function only returns two types of cliques: fully connected
 * (NVSwitch) or HCM, to support typically GPU topologies.  In all
 * other cases, we return an empty vector.
 *
 * We do not currently account for the number of NVLinks connecting
 * two GPUs because our algorithm does not take advantage of that
 * (yet!).
 *
 * This also enables p2p access between all GPUs that are connected
 * using NVLink.  Unfortunately, we do not have a good way to detect
 * which GPUs are being used for this particular job, so we enable p2p
 * access for all connected GPUs.
 */
static ncclResult_t topoDetect(
    ncclComm_t comm,
    std::vector<std::vector<int>>& cliques_) {
  int nGPUs;
  std::vector<std::vector<int>> cliques;
  CUDACHECK(cudaGetDeviceCount(&nGPUs));

  /* perf rank Matrix is like an adjacency matrix, but ranks links
   * based on performance.  Rank 0 means very fast connectivity. */
  int perfRankMatrix[nGPUs][nGPUs];
  uint8_t adjacencyMatrix[nGPUs][nGPUs];

  for (int i = 0; i < nGPUs; i++) {
    for (int j = 0; j < nGPUs; j++) {
      if (i == j) {
        perfRankMatrix[i][j] = 0;
      } else if (ncclParamForceP2pAccess()) {
        perfRankMatrix[i][j] = 0;
      } else {
        int val;
        CUDACHECK(cudaDeviceGetP2PAttribute(&val, cudaDevP2PAttrPerformanceRank, i, j));
        perfRankMatrix[i][j] = val;
      }
    }
  }

  /* set adjacency matrix */
  for (int i = 0; i < nGPUs; i++) {
    for (int j = 0; j < nGPUs; j++) {
      if (perfRankMatrix[i][j] < 2) {
        adjacencyMatrix[i][j] = 1;
      } else {
        adjacencyMatrix[i][j] = 0;
      }
    }
  }


  /***** Detect fully connected (NVSwitch) topology *****/
  bool connected;
  connected = true;
  for (int i = 0; i < nGPUs; i++) {
    for (int j = 0; j < nGPUs; j++) {
      if (adjacencyMatrix[i][j] == 0) {
        connected = false;
        break;
      }
    }
    if (connected == false) {
      break;
    }
  }
  if (connected) {
    std::vector<int> v;
    for (int i = 0; i < nGPUs; i++) {
      v.push_back(i);
    }
    cliques.push_back(v);
    goto exit;
  }

  /***** Detect HCM topology *****/
  for (int i = 0; i < nGPUs; i++) {
    cliques.push_back(std::vector<int>{i});
  }

  /* find cliques of size nGPUs/2 */
  for (int k = 2; k <= nGPUs / 2; k++) {
    std::vector<std::vector<int>> tmp;
    for (auto v : cliques) {
      for (int i = v.back() + 1; i < nGPUs; i++) {
        bool connected = true;
        for (auto j : v) {
          if (adjacencyMatrix[i][j] == 0) {
            connected = false;
            break;
          }
        }
        if (connected) {
          std::vector<int> w = v;
          w.push_back(i);
          tmp.push_back(w);
        }
      }
    }

    cliques.clear();
    cliques = tmp;
    tmp.clear();
  }

  /* HCM has two cliques */
  if (cliques.size() != 2) {
    goto topo_not_found;
  }

  /* In HCM, each GPU is connected to (nGPUs/2 + 1) other GPUs */
  for (int i = 0; i < nGPUs; i++) {
    int count = 0;
    for (int j = 0; j < nGPUs; j++) {
      if (adjacencyMatrix[i][j]) {
        count++;
      }
    }
    if (count != (1 + nGPUs / 2)) {
      goto topo_not_found;
    }
  }

  /* layout the cliques, so the two HCM cliques are ordered so that
   * the i'th GPU in one clique is connected to the i'th GPU in the
   * other clique.  If this is not possible, it's not HCM. */
  {
    std::vector<int> front = cliques.front();
    std::vector<int> back = cliques.back();
    std::vector<int> tmp;
    for (auto f : front) {
      /* each element in front should be connected to exactly
       * one element in back */
      for (auto b : back) {
        if (adjacencyMatrix[f][b]) {
          tmp.push_back(b);
          break;
        }
      }
    }
    assert(tmp.size() == nGPUs / 2);
    cliques.pop_back();
    cliques.push_back(tmp);
  }

exit:
  for (int i = 0; i < nGPUs; i++) {
    int dev;
    CUDACHECK(cudaGetDevice(&dev));
    if (i == dev) {
      continue;
    } else if (adjacencyMatrix[comm->cudaDev][i] == 1) {
      cudaError_t e = cudaDeviceEnablePeerAccess(i, 0);
      if (e != cudaErrorPeerAccessAlreadyEnabled && e != cudaSuccess) {
        CUDACHECK(e);
      }
    }
  }
  cliques_ = cliques;
  return ncclSuccess;

topo_not_found:
  return ncclSuccess;
}

/*
 * This is our core function to detect the number of ranks in the same
 * virtual address space (i.e., threads).  The first communicator
 * handle that is created with a particular context ID creates and
 * enqueues an ddaThreadSharedMd object in the ddaThreadSharedMdList
 * queue.  If a new communicator handle is created with the same
 * context ID, it would point to the same ddaThreadSharedMd object.  The
 * number of communicator handles pointing to the ddaThreadSharedMd
 * object determines the number of dda ranks in this address
 * space.
 */
ncclResult_t allocDDAMd(ncclComm *comm, ncclUniqueId commId) {
  // set enableIpc flag
  char* allreduceAlgoStr = getenv("NCCL_ALLREDUCE_ALGO");
  const bool enableIpc =
      (allreduceAlgoStr != nullptr &&
       !strcmp(allreduceAlgoStr, "dda_ipc"));

  ddaThreadSharedMd* threadSharedMd;
  ncclResult_t ret = ncclSuccess;
  std::vector<std::vector<int>> gpuCliques;

  // IPC states start
  // TODO: variables should be declared close to their usage, but can't be
  // done here due to NCCLCHECKGOTO, need a fix later
  const size_t kNumHandles = 3;
  cudaIpcMemHandle_t localHdls[kNumHandles];
  void* handleSendBuf{nullptr};
  void* handleRecvBuf{nullptr};
  const size_t kHandleSize = sizeof(cudaIpcMemHandle_t);
  const size_t kBarrierSize =
      2 * comm->nRanks * comm->nRanks * sizeof(uintptr_t);
  cudaStream_t stream;
  cudaIpcMemHandle_t allHdls[comm->nRanks * kNumHandles];
  // IPC states end

  NCCLCHECKGOTO(topoDetect(comm, gpuCliques), ret, exit);

  ddaThreadSharedMdListMutex.lock();

  /* allocate the ddaThreadSharedMd structure or find an existing
   * one for this commId */
  threadSharedMd = nullptr;
  for (auto t : ddaThreadSharedMdList) {
    if (t->commId == commId) {
      threadSharedMd = t;
      break;
    }
  }
  if (threadSharedMd == nullptr) {
    threadSharedMd = new ddaThreadSharedMd(commId, gpuCliques, enableIpc);
    ddaThreadSharedMdList.push_back(threadSharedMd);
  }

  threadSharedMd->insertRank(comm->rank, comm->cudaDev);
  threadSharedMd->refCount++;

  comm->dda = new ddaPrivateMd(threadSharedMd, comm->cudaDev, gpuCliques.size());

  ddaThreadSharedMdListMutex.unlock();

  if (threadSharedMd->enableIpc()) {
    // allocate dev mem
    CUDACHECK(cudaMalloc(&comm->dda->barrierMbox[0], kBarrierSize));
    CUDACHECK(cudaMalloc(&comm->dda->barrierMbox[1], kBarrierSize));
    CUDACHECK(cudaMalloc(&comm->dda->allTmpSendbuffs, comm->nRanks * sizeof(uintptr_t)));

    // allocate host mem
    comm->dda->allTmpSendbuffsHost =
        static_cast<void**>(malloc(comm->nRanks * sizeof(uintptr_t)));

    // open local handles
    CUDACHECK(cudaIpcGetMemHandle(&localHdls[0], comm->dda->barrierMbox[0]));
    CUDACHECK(cudaIpcGetMemHandle(&localHdls[1], comm->dda->barrierMbox[1]));
    CUDACHECK(cudaIpcGetMemHandle(&localHdls[2], comm->dda->tmpbuff));

    // copy handles to local sendBuf
    CUDACHECK(cudaMalloc(&handleSendBuf, kHandleSize * kNumHandles));
    CUDACHECK(
        cudaMalloc(&handleRecvBuf, kHandleSize * comm->nRanks * kNumHandles));
    CUDACHECK(cudaMemcpy(
        handleSendBuf,
        &localHdls,
        kHandleSize * kNumHandles,
        cudaMemcpyDefault));

    // all gather local handles
    cudaStreamCreate(&stream);
    NCCLCHECK(ncclAllGather(
        handleSendBuf,
        handleRecvBuf,
        kHandleSize * kNumHandles,
        ncclUint8,
        comm,
        stream));
    CUDACHECK(cudaStreamSynchronize(stream));

    // deserialize all hanles
    CUDACHECK(cudaMemcpy(
        allHdls,
        handleRecvBuf,
        kHandleSize * comm->nRanks * kNumHandles,
        cudaMemcpyDefault));

    // all gather completed, free send/recv buf
    CUDACHECK(cudaFree(handleSendBuf));
    CUDACHECK(cudaFree(handleRecvBuf));

    // update comm->dda->allSend/Recv TmpBufs[nRanks]
    for (size_t rankIdx = 0; rankIdx < comm->nRanks; ++rankIdx) {
      const auto& barrierHdl0 = allHdls[rankIdx * kNumHandles];
      const auto& barrierHdl1 = allHdls[rankIdx * kNumHandles + 1];
      const auto& sendBufHdl = allHdls[rankIdx * kNumHandles + 2];
      if (comm->rank == rankIdx) {
        // local rank should point to local buf
        comm->dda->allTmpSendbuffsHost[rankIdx] = comm->dda->tmpbuff;
      } else {
        // otherwise, open IPC handle
        void* remoteBuf = nullptr;
        CUDACHECK(cudaIpcOpenMemHandle(
            (void**)&remoteBuf, sendBufHdl, cudaIpcMemLazyEnablePeerAccess));
        comm->dda->allTmpSendbuffsHost[rankIdx] = remoteBuf;
      }
    }
    CUDACHECK(cudaMemcpy(
        comm->dda->allTmpSendbuffs,
        comm->dda->allTmpSendbuffsHost,
        comm->nRanks * sizeof(uintptr_t),
        cudaMemcpyDefault));

    // update comm->dda->barrierMbox, all ranks should use rank0's barrier
    // TODO should use lowest rank ID instead?
    if (comm->rank != 0) {
      void* remoteBuf = nullptr;
      CUDACHECK(cudaIpcOpenMemHandle(
          (void**)&remoteBuf, allHdls[0], cudaIpcMemLazyEnablePeerAccess));
      comm->dda->barrierMbox[0] = reinterpret_cast<uintptr_t*>(remoteBuf);

      remoteBuf = nullptr;
      CUDACHECK(cudaIpcOpenMemHandle(
          (void**)&remoteBuf, allHdls[1], cudaIpcMemLazyEnablePeerAccess));
      comm->dda->barrierMbox[1] = reinterpret_cast<uintptr_t*>(remoteBuf);
    }
  }
exit:
  return ret;
}

/* This function decreases the refCount for the ddaThreadSharedMd object
 * (one of the communicator pointing to it is getting freed).  If the
 * refCount reaches zero, that means no communicators are pointing to
 * it -- in that case, we can remove it from the
 * ddaThreadSharedMdList. */
ncclResult_t freeDDAMd(ncclComm *comm) {
  ddaThreadSharedMd *threadSharedMd = comm->dda->threadSharedMd;

  ddaThreadSharedMdListMutex.lock();

  threadSharedMd->refCount--;

  if (threadSharedMd->refCount == 0) {
    if (threadSharedMd->enableIpc()) {
      // close ipc handles
      for (int i = 0; i < comm->nRanks; ++i) {
        if (i == comm->rank) {
          continue;
        }
        CUDACHECKIGNORE(cudaIpcCloseMemHandle(comm->dda->allTmpSendbuffsHost[i]));
      }
      // free host/dev memories
      CUDACHECKIGNORE(cudaFree(comm->dda->barrierMbox[0]));
      CUDACHECKIGNORE(cudaFree(comm->dda->barrierMbox[1]));
      CUDACHECKIGNORE(cudaFree(comm->dda->allTmpSendbuffs));
    }

    auto threadSharedMdIdx =
        std::remove(ddaThreadSharedMdList.begin(), ddaThreadSharedMdList.end(), threadSharedMd);
    ddaThreadSharedMdList.erase(threadSharedMdIdx, ddaThreadSharedMdList.end());
    delete threadSharedMd;
  }

  ddaThreadSharedMdListMutex.unlock();
  delete comm->dda;

  return ncclSuccess;
}
